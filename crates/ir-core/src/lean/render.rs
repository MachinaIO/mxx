use super::model::{LEAN_IR_VERSION, LeanEmissionError, RenderedLeanModule, RenderedLeanProgram};
use crate::{
    encoding::hash_canonical,
    linked::{
        ChildInputHop, ConcreteLinkedProgram, ConcreteLinkedStage, ConcreteMatrixLiteral,
        ConcreteNode, ConcreteNodePayload, ConcreteWireRef, ParallelOutputHop,
        StructuralValueRoute, ValidatedLinkedProgram,
    },
    types::ConcreteWireType,
};
use num_bigint::BigInt;
use std::{collections::BTreeSet, fmt::Write, path::PathBuf};

const CERTIFICATE_CHUNK_SIZE: usize = 16;
const NODE_CERTIFICATE_MODULE_TARGET_BYTES: usize = 512 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
struct RangeCertificateExpr {
    start: usize,
    end: usize,
    depth: usize,
    source: String,
}

fn balanced_range_expression(
    constructor: &str,
    leaves: &[RangeCertificateExpr],
    empty_at: usize,
) -> RangeCertificateExpr {
    if leaves.is_empty() {
        return RangeCertificateExpr {
            start: empty_at,
            end: empty_at,
            depth: 1,
            source: format!("{constructor}.empty {empty_at}"),
        };
    }
    if leaves.len() == 1 {
        return leaves[0].clone();
    }
    let middle = leaves.len() / 2;
    let left = balanced_range_expression(constructor, &leaves[..middle], empty_at);
    let right = balanced_range_expression(constructor, &leaves[middle..], left.end);
    debug_assert_eq!(left.end, right.start);
    RangeCertificateExpr {
        start: left.start,
        end: right.end,
        depth: 1 + left.depth.max(right.depth),
        source: format!("{constructor}.append ({}) ({})", left.source, right.source),
    }
}

fn certificate_chunks(constructor: &str, leaf_names: &[String]) -> Vec<RangeCertificateExpr> {
    leaf_names
        .chunks(CERTIFICATE_CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let start = chunk_index * CERTIFICATE_CHUNK_SIZE;
            let leaves = chunk
                .iter()
                .enumerate()
                .map(|(offset, name)| {
                    let index = start + offset;
                    RangeCertificateExpr {
                        start: index,
                        end: index + 1,
                        depth: 1,
                        source: format!("{constructor}.single {index} {name}"),
                    }
                })
                .collect::<Vec<_>>();
            balanced_range_expression(constructor, &leaves, start)
        })
        .collect()
}

fn render_named_range(
    source: &mut String,
    name: &str,
    type_prefix: &str,
    constructor: &str,
    leaf_names: &[String],
) {
    let chunks = certificate_chunks(constructor, leaf_names);
    let mut chunk_names = Vec::with_capacity(chunks.len());
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let chunk_name = format!("{name}Chunk{chunk_index}");
        chunk_names.push(RangeCertificateExpr {
            start: chunk.start,
            end: chunk.end,
            depth: 1,
            source: chunk_name.clone(),
        });
        writeln!(
            source,
            "def {chunk_name} : {type_prefix} {} {} :=\n  {}\n",
            chunk.start, chunk.end, chunk.source
        )
        .unwrap();
    }
    let complete = balanced_range_expression(constructor, &chunk_names, 0);
    writeln!(
        source,
        "def {name} : {type_prefix} 0 {} :=\n  {}\n",
        leaf_names.len(),
        complete.source
    )
    .unwrap();
}

#[allow(clippy::too_many_arguments)]
fn render_optional_types_certificate(
    source: &mut String,
    name: &str,
    actual: &str,
    expected: &str,
    actual_fact: &str,
    actual_canonical: &str,
    expected_fact: &str,
    expected_canonical: &str,
    actual_types: &[&ConcreteWireType],
    expected_types: &[&ConcreteWireType],
) -> Result<(), LeanEmissionError> {
    if actual_types.len() != expected_types.len() {
        return Err(LeanEmissionError::Encoding {
            message: format!("optional type certificate {name} has unequal lengths"),
        });
    }
    let mut leaves = Vec::with_capacity(actual_types.len());
    for (index, (actual_type, expected_type)) in actual_types.iter().zip(expected_types).enumerate()
    {
        let compatibility = render_type_compatibility(actual_type, expected_type)?;
        let leaf = format!("{name}Entry{index}");
        leaves.push(leaf.clone());
        writeln!(
            source,
            "def {leaf} : Mxx.IR.OptionalTypePairCert ({actual}) ({expected}) {index} where\n  actualType := {}\n  expectedType := {}\n  actualStored := (congrArg (fun values => values[{index}]?) {actual_fact}).trans (by rfl)\n  expectedStored := (congrArg (fun values => values[{index}]?) {expected_fact}).trans (by rfl)\n  compatible := {compatibility}\n",
            render_type(actual_type)?,
            render_type(expected_type)?
        )
        .unwrap();
    }
    let range = format!("{name}Entries");
    render_named_range(
        source,
        &range,
        &format!("Mxx.IR.DataRangeCert (Mxx.IR.OptionalTypePairCert ({actual}) ({expected}))"),
        "Mxx.IR.DataRangeCert",
        &leaves,
    );
    writeln!(
        source,
        "def {name} : Mxx.IR.OptionalTypesCert ({actual}) ({expected}) where\n  lengthEq := by\n    calc\n      ({actual}).length = ({actual_canonical}).length := congrArg List.length {actual_fact}\n      _ = ({expected_canonical}).length := by rfl\n      _ = ({expected}).length := (congrArg List.length {expected_fact}).symm\n  entries := {range}\n"
    )
    .unwrap();
    Ok(())
}

fn render_type_compatibility(
    actual: &ConcreteWireType,
    expected: &ConcreteWireType,
) -> Result<String, LeanEmissionError> {
    if actual == expected {
        return Ok(format!("Mxx.IR.TypeCompatibilityCert.exact ({})", render_type(actual)?));
    }
    match (actual, expected) {
        (ConcreteWireType::ConstantInt, ConcreteWireType::Int) => {
            Ok("Mxx.IR.TypeCompatibilityCert.constantInt".to_owned())
        }
        (ConcreteWireType::ConstantBool, ConcreteWireType::Bool) => {
            Ok("Mxx.IR.TypeCompatibilityCert.constantBool".to_owned())
        }
        (ConcreteWireType::ConstantReal, ConcreteWireType::Real) => {
            Ok("Mxx.IR.TypeCompatibilityCert.constantReal".to_owned())
        }
        _ => Err(LeanEmissionError::Encoding {
            message: format!("incompatible structural wire types: {actual:?} and {expected:?}"),
        }),
    }
}

fn render_nat_list(values: &[usize]) -> String {
    format!("[{}]", values.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
}

fn render_index_map_checked_proof(map: &crate::linked::ConcreteIndexMap) -> String {
    let size = map.input_indices.len();
    let bounded = if size == 0 {
        "omega".to_owned()
    } else if size == 1 {
        "change x < 1 at bound\n          have : x = 0 := by omega\n          subst x\n          simp [Mxx.IR.indexExprFuel, Mxx.IR.indexExprCheckedFuelB, Mxx.IR.indexSlotAllowedB, Mxx.IR.checkedIndexBinary]"
            .to_owned()
    } else {
        let cases = (0..size).map(|index| format!("x = {index}")).collect::<Vec<_>>().join(" ∨ ");
        let patterns = std::iter::repeat_n("rfl", size).collect::<Vec<_>>().join(" | ");
        let branches = std::iter::repeat_n(
            "          · simp [Mxx.IR.indexExprFuel, Mxx.IR.indexExprCheckedFuelB, Mxx.IR.indexSlotAllowedB, Mxx.IR.checkedIndexBinary]",
            size,
        )
        .collect::<Vec<_>>()
        .join("\n");
        format!(
            "change x < {size} at bound\n          have cases : {cases} := by omega\n          rcases cases with {patterns}\n{branches}"
        )
    };
    format!(
        "by\n        simp only [Mxx.IR.indexMapCheckedB, Bool.and_eq_true, decide_eq_true_eq,\n          List.all_eq_true, List.mem_range]\n        constructor\n        · rfl\n        · intro x bound\n          {bounded}"
    )
}

fn scope_ranks(
    stage: &crate::linked::ConcreteLinkedStage,
) -> Result<Vec<usize>, LeanEmissionError> {
    fn visit(
        stage: &crate::linked::ConcreteLinkedStage,
        index: usize,
        states: &mut [u8],
        ranks: &mut [usize],
    ) -> Result<usize, LeanEmissionError> {
        match states.get(index).copied() {
            Some(2) => return Ok(ranks[index]),
            Some(1) => {
                return Err(LeanEmissionError::Encoding {
                    message: format!(
                        "stage {:?} contains a structural child cycle at scope {index}",
                        stage.key
                    ),
                });
            }
            Some(0) => {}
            None => {
                return Err(LeanEmissionError::Encoding {
                    message: format!(
                        "stage {:?} refers to missing structural child scope {index}",
                        stage.key
                    ),
                });
            }
            Some(_) => unreachable!(),
        }
        states[index] = 1;
        let mut rank = 0;
        for child in stage.scopes[index].nodes.iter().filter_map(|node| node.child_scope) {
            rank = rank.max(visit(stage, child, states, ranks)? + 1);
        }
        states[index] = 2;
        ranks[index] = rank;
        Ok(rank)
    }

    let mut states = vec![0; stage.scopes.len()];
    let mut ranks = vec![0; stage.scopes.len()];
    for index in 0..stage.scopes.len() {
        visit(stage, index, &mut states, &mut ranks)?;
    }
    Ok(ranks)
}

/// Render one complete `Mxx.IR.ProgramData` from the validated semantic AST.
pub fn render_lean_program(
    program: &ValidatedLinkedProgram,
    module_root: &str,
) -> Result<RenderedLeanProgram, LeanEmissionError> {
    validate_module_name(module_root)?;
    let ast = program
        .semantic_projection()
        .map_err(|error| LeanEmissionError::Encoding { message: error.to_string() })?;
    let hash = hash_canonical(&ast)
        .map_err(|error| LeanEmissionError::Encoding { message: error.to_string() })?;
    let mut data = String::from(
        "import MxxIrCore.Program\n\nset_option linter.unusedSimpArgs false\n\nnamespace Mxx.Generated\n\n",
    );
    writeln!(data, "def linkedProgramSha256 : List UInt8 := {}", bytes(&hash)).unwrap();
    writeln!(data, "def irVersion : Nat := {}\n", LEAN_IR_VERSION).unwrap();
    for (stage_index, stage) in ast.stages.iter().enumerate() {
        for scope in &stage.scopes {
            for (node_index, node) in scope.nodes.iter().enumerate() {
                let payload = render_payload(node, node_index, stage, stage_index, scope.id, &ast)?;
                writeln!(
                    data,
                    "abbrev stage{stage_index}_scope{}_node{node_index} : Mxx.IR.Node := {{",
                    scope.id
                )
                .unwrap();
                writeln!(data, "  payload := {payload},").unwrap();
                writeln!(
                    data,
                    "  arguments := #[{}],",
                    node.arguments.iter().map(render_wire).collect::<Vec<_>>().join(", ")
                )
                .unwrap();
                writeln!(
                    data,
                    "  outputs := #[{}]",
                    node.outputs.iter().map(render_type).collect::<Result<Vec<_>, _>>()?.join(", ")
                )
                .unwrap();
                writeln!(data, "}}\n").unwrap();
            }
            writeln!(data, "abbrev stage{}_scope{} : Mxx.IR.Scope := {{", stage_index, scope.id)
                .unwrap();
            writeln!(data, "  id := {},", scope.id).unwrap();
            writeln!(
                data,
                "  structuralSlots := #[{}],",
                scope
                    .structural_slots
                    .iter()
                    .map(render_structural_slot)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                data,
                "  nodes := #[{}],",
                (0..scope.nodes.len())
                    .map(|node_index| {
                        format!("stage{stage_index}_scope{}_node{node_index}", scope.id)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();
            writeln!(
                data,
                "  inputs := #[{}],",
                scope.inputs.iter().map(render_wire).collect::<Vec<_>>().join(", ")
            )
            .unwrap();
            writeln!(
                data,
                "  outputs := #[{}]",
                scope.outputs.iter().map(render_wire).collect::<Vec<_>>().join(", ")
            )
            .unwrap();
            writeln!(data, "}}\n").unwrap();
            writeln!(
                data,
                "def stage{stage_index}_scope{}StructuralSlots : stage{stage_index}_scope{}.structuralSlots = #[{}] := by rfl\n",
                scope.id,
                scope.id,
                scope
                    .structural_slots
                    .iter()
                    .map(render_structural_slot)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .unwrap();
        }
        writeln!(data, "abbrev stage{} : Mxx.IR.Stage := {{", stage_index).unwrap();
        writeln!(data, "  name := {},", lean_string(&stage.key)).unwrap();
        writeln!(
            data,
            "  bindings := #[{}],",
            stage
                .bindings
                .integers
                .iter()
                .map(|(name, value)| format!("({}, {})", lean_string(name), value))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(
            data,
            "  scopes := #[{}],",
            stage
                .scopes
                .iter()
                .map(|scope| format!("stage{}_scope{}", stage_index, scope.id))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(data, "  root := {},", stage.root_scope).unwrap();
        writeln!(
            data,
            "  namedOutputs := #[{}]",
            stage
                .named_outputs
                .iter()
                .map(|output| format!(
                    "{{ name := {}, wire := {} }}",
                    lean_string(&output.name),
                    render_wire(&output.wire)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(data, "}}\n").unwrap();
    }
    for (link_index, link) in ast.artifact_links.iter().enumerate() {
        writeln!(
            data,
            "abbrev artifactLink{link_index} : Mxx.IR.ArtifactLink := {{ consumerStage := {}, consumer := {}, argument := {}, consumerArtifact := {}, consumerConfidentiality := {}, consumerType := {}, producerStage := {}, producer := {}, producerArtifact := {}, producerConfidentiality := {}, producerType := {} }}\n",
            link.consumer_stage,
            render_wire(&link.consumer),
            link.argument,
            lean_string(&link.consumer_artifact),
            render_confidentiality(link.consumer_confidentiality),
            render_type(&link.consumer_type)?,
            link.producer_stage,
            render_wire(&link.producer),
            lean_string(&link.producer_artifact),
            render_confidentiality(link.producer_confidentiality),
            render_type(&link.producer_type)?
        )
        .unwrap();
    }
    writeln!(
        data,
        "abbrev artifactLinks : Array Mxx.IR.ArtifactLink := #[{}]\n",
        (0..ast.artifact_links.len())
            .map(|link_index| format!("artifactLink{link_index}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
    .unwrap();
    writeln!(data, "abbrev linkedProgramData : Mxx.IR.ProgramData := {{").unwrap();
    writeln!(
        data,
        "  identity := {{ irVersion := irVersion, linkedProgramSha256 := linkedProgramSha256 }},"
    )
    .unwrap();
    writeln!(
        data,
        "  stages := #[{}],",
        (0..ast.stages.len()).map(|i| format!("stage{}", i)).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    writeln!(data, "  artifactLinks := artifactLinks\n}}\n").unwrap();
    render_scope_lookup_certificates(&mut data, &ast)?;
    writeln!(data, "end Mxx.Generated").unwrap();

    let data_module = format!("{module_root}.Data");
    let mut modules = vec![rendered_module(data_module.clone(), data)];
    let scope_sources = render_node_certificate_scopes(&ast)?;
    let mut chunk_names = Vec::new();
    for chunk_body in chunk_scope_sources(scope_sources, NODE_CERTIFICATE_MODULE_TARGET_BYTES) {
        let chunk_name = format!("{module_root}.NodeCerts{:02}", chunk_names.len());
        modules.push(rendered_module(
            chunk_name.clone(),
            module_with_namespace(&[&data_module], &chunk_body),
        ));
        chunk_names.push(chunk_name);
    }

    // Equation modules are generated from the same concrete node aliases as the certificate.
    // They import only Data and the checked evaluator, so a consumer can use the equations
    // without importing a monolithic proof file.  A scope is kept intact while chunking.
    let equation_sources = render_node_equation_scopes(&ast)?;
    let mut equation_names = Vec::new();
    for chunk_body in chunk_scope_sources(equation_sources, NODE_CERTIFICATE_MODULE_TARGET_BYTES) {
        let equation_name = format!("{module_root}.NodeEquations{:02}", equation_names.len());
        modules.push(rendered_module(
            equation_name.clone(),
            module_with_imports(&[&data_module, "MxxIrCore.ScopeInvariant"], &chunk_body),
        ));
        equation_names.push(equation_name);
    }

    let certificate_module = format!("{module_root}.Certificate");
    let mut certificate_body = String::new();
    render_program_certificate(&mut certificate_body, &ast)?;
    let mut certificate_imports = Vec::with_capacity(chunk_names.len() + 1);
    certificate_imports.push(data_module.as_str());
    certificate_imports.extend(chunk_names.iter().map(String::as_str));
    modules.push(rendered_module(
        certificate_module.clone(),
        module_with_namespace(&certificate_imports, &certificate_body),
    ));
    // These wrappers reconnect the public evaluator API to every concrete stage index.  The
    // certificate is imported here (rather than by NodeEquations) so equation modules remain
    // usable as local evaluator-step libraries without creating an import cycle.
    let stage_root_sources = render_stage_root_equations(&ast)?;
    let mut stage_root_names = Vec::new();
    for chunk_body in chunk_scope_sources(stage_root_sources, NODE_CERTIFICATE_MODULE_TARGET_BYTES)
    {
        let stage_root_name = format!("{module_root}.StageRoots{:02}", stage_root_names.len());
        let mut stage_root_imports = vec![certificate_module.as_str(), "MxxIrCore.NodeEquation"];
        stage_root_imports.extend(equation_names.iter().map(String::as_str));
        modules.push(rendered_module(
            stage_root_name.clone(),
            module_with_imports(&stage_root_imports, &chunk_body),
        ));
        stage_root_names.push(stage_root_name);
    }
    let mut root_source = format!("import {certificate_module}\n");
    for equation_name in &equation_names {
        writeln!(root_source, "import {equation_name}").unwrap();
    }
    for stage_root_name in &stage_root_names {
        writeln!(root_source, "import {stage_root_name}").unwrap();
    }
    modules.push(rendered_module(module_root.to_owned(), root_source));
    validate_module_order(module_root, &modules)?;
    Ok(RenderedLeanProgram {
        modules,
        root_module: module_root.to_owned(),
        ir_version: LEAN_IR_VERSION,
        linked_program_sha256: hash,
    })
}

fn chunk_scope_sources(scopes: Vec<(String, String)>, target_bytes: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for (_scope_name, scope) in scopes {
        if !current.is_empty() && current.len() + scope.len() > target_bytes {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(&scope);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn validate_module_name(module: &str) -> Result<(), LeanEmissionError> {
    let valid = !module.is_empty() &&
        module.split('.').all(|part| {
            let mut characters = part.chars();
            characters.next().is_some_and(|first| first.is_ascii_alphabetic()) &&
                characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
        });
    if valid {
        Ok(())
    } else {
        Err(LeanEmissionError::Encoding { message: format!("invalid Lean module root {module:?}") })
    }
}

fn rendered_module(module_name: String, source: String) -> RenderedLeanModule {
    let relative_path = PathBuf::from(module_name.replace('.', "/") + ".lean");
    RenderedLeanModule { module_name, relative_path, source }
}

fn validate_module_order(
    module_root: &str,
    modules: &[RenderedLeanModule],
) -> Result<(), LeanEmissionError> {
    let mut names = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for module in modules {
        if names.contains(module.module_name.as_str()) ||
            !paths.insert(module.relative_path.as_path())
        {
            return Err(LeanEmissionError::Encoding {
                message: "generated Lean modules are not unique".to_owned(),
            });
        }
        for import in module
            .source
            .lines()
            .filter_map(|line| line.strip_prefix("import "))
            .filter(|import| import.starts_with(module_root))
        {
            if !names.contains(import) {
                return Err(LeanEmissionError::Encoding {
                    message: format!(
                        "generated module {} imports later or missing module {import}",
                        module.module_name
                    ),
                });
            }
        }
        names.insert(module.module_name.as_str());
    }
    Ok(())
}

fn module_with_namespace(imports: &[&str], body: &str) -> String {
    let mut source = String::new();
    for import in imports {
        writeln!(source, "import {import}").unwrap();
    }
    source.push_str("\nset_option linter.unusedSimpArgs false\n\nnamespace Mxx.Generated\n\n");
    source.push_str(body);
    source.push_str("end Mxx.Generated\n");
    source
}

fn module_with_imports(imports: &[&str], body: &str) -> String {
    let mut source = String::new();
    for import in imports {
        writeln!(source, "import {import}").unwrap();
    }
    source.push_str("\nset_option linter.unusedSimpArgs false\n\nnamespace Mxx.Generated\n\n");
    source.push_str(body);
    source.push_str("end Mxx.Generated\n");
    source
}

fn render_scope_lookup_certificates(
    source: &mut String,
    ast: &ConcreteLinkedProgram,
) -> Result<(), LeanEmissionError> {
    for (stage_index, stage) in ast.stages.iter().enumerate() {
        for scope in &stage.scopes {
            writeln!(
                source,
                "def stage{stage_index}_scope{}Stored : Mxx.IR.scopeAt stage{stage_index} {} = some stage{stage_index}_scope{} := by\n  simp [stage{stage_index}, Mxx.IR.scopeAt]\n",
                scope.id, scope.id, scope.id
            )
            .unwrap();
            for (label, wires) in [("Input", &scope.inputs), ("Output", &scope.outputs)] {
                let types = wires
                    .iter()
                    .map(|wire| resolve_scope_wire_type(scope, wire))
                    .map(|result| {
                        result.and_then(|wire_type| {
                            render_type(wire_type).map(|ty| format!("some ({ty})"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                writeln!(
                    source,
                    "abbrev stage{stage_index}_scope{}{label}Types : List (Option Mxx.IR.WireType) := [{types}]",
                    scope.id
                )
                .unwrap();
                writeln!(
                    source,
                    "def stage{stage_index}_scope{}{label}sTyped : Mxx.IR.referencedTypes stage{stage_index}_scope{} stage{stage_index}_scope{}.{label_lower}s = stage{stage_index}_scope{}{label}Types := by rfl\n",
                    scope.id,
                    scope.id,
                    scope.id,
                    scope.id,
                    label_lower = label.to_ascii_lowercase()
                )
                .unwrap();
            }
        }
    }
    Ok(())
}

fn resolve_scope_wire_type<'a>(
    scope: &'a crate::linked::ConcreteScope,
    wire: &ConcreteWireRef,
) -> Result<&'a ConcreteWireType, LeanEmissionError> {
    if wire.scope != scope.id {
        return Err(LeanEmissionError::Encoding {
            message: format!("scope {} contains a wire owned by scope {}", scope.id, wire.scope),
        });
    }
    scope
        .nodes
        .get(wire.node.0 as usize)
        .and_then(|node| node.outputs.get(wire.port.0 as usize))
        .ok_or_else(|| LeanEmissionError::Encoding {
            message: format!(
                "scope {} wire {}:{} does not resolve",
                scope.id, wire.node.0, wire.port.0
            ),
        })
}

fn alternatives(items: impl IntoIterator<Item = String>) -> String {
    items.into_iter().collect::<Vec<_>>().join(" ∨ ")
}

fn render_scope_slots_proof(scope_name: &str, scope: &crate::linked::ConcreteScope) -> String {
    let count = scope.structural_slots.len();
    let pairs = (0..count)
        .flat_map(|first| {
            (0..count)
                .filter(move |second| *second != first)
                .map(move |second| format!("(first = {first} ∧ second = {second})"))
        })
        .collect::<Vec<_>>();
    let mut proof = String::from(
        "by\n      constructor\n      · intro first second left right leftStored rightStored different\n        rcases Array.getElem?_eq_some_iff.mp leftStored with ⟨leftBound, leftEq⟩\n        rcases Array.getElem?_eq_some_iff.mp rightStored with ⟨rightBound, rightEq⟩\n",
    );
    writeln!(proof, "        change first < {count} at leftBound").unwrap();
    writeln!(proof, "        change second < {count} at rightBound").unwrap();
    if pairs.is_empty() {
        proof.push_str("        omega\n");
    } else {
        writeln!(proof, "        have cases : {} := by omega", alternatives(pairs)).unwrap();
        let patterns = (0..count)
            .flat_map(|first| {
                (0..count).filter(move |second| *second != first).map(|_| "⟨rfl, rfl⟩".to_owned())
            })
            .collect::<Vec<_>>();
        writeln!(proof, "        rcases cases with {}", patterns.join(" | ")).unwrap();
        for _ in &patterns {
            proof.push_str(
                "        · simp at leftEq rightEq\n          subst left\n          subst right\n          decide\n",
            );
        }
    }
    proof.push_str("      · intro slot member\n");
    if count == 0 {
        writeln!(proof, "        simp [{scope_name}] at member").unwrap();
    } else {
        let declarations = scope
            .structural_slots
            .iter()
            .map(|slot| format!("slot = {}", render_structural_slot(slot)));
        writeln!(
            proof,
            "        have cases : {} := by simpa [{scope_name}] using member",
            alternatives(declarations)
        )
        .unwrap();
        writeln!(
            proof,
            "        rcases cases with {}",
            std::iter::repeat_n("rfl", count).collect::<Vec<_>>().join(" | ")
        )
        .unwrap();
        for _ in 0..count {
            proof.push_str("        · decide\n");
        }
    }
    proof.trim_end().to_owned()
}

fn render_scope_wires_proof(
    scope_name: &str,
    scope: &crate::linked::ConcreteScope,
    wires: &[ConcreteWireRef],
) -> Result<String, LeanEmissionError> {
    let mut proof = String::from("by\n      intro wire member\n");
    if wires.is_empty() {
        writeln!(proof, "      simp [{scope_name}] at member").unwrap();
        return Ok(proof.trim_end().to_owned());
    }
    writeln!(
        proof,
        "      have cases : {} := by simpa [{scope_name}] using member",
        alternatives(wires.iter().map(|wire| format!("wire = {}", render_wire(wire))))
    )
    .unwrap();
    writeln!(
        proof,
        "      rcases cases with {}",
        std::iter::repeat_n("rfl", wires.len()).collect::<Vec<_>>().join(" | ")
    )
    .unwrap();
    for wire in wires {
        let wire_type = resolve_scope_wire_type(scope, wire)?;
        writeln!(proof, "      · exact ⟨rfl, {}, by rfl⟩", render_type(wire_type)?).unwrap();
    }
    Ok(proof.trim_end().to_owned())
}

fn first_occurrence_unique(items: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut unique = Vec::new();
    for item in items {
        if !unique.contains(&item) {
            unique.push(item);
        }
    }
    unique
}

fn render_children_decrease_proof(
    stage_index: usize,
    scope_name: &str,
    scope: &crate::linked::ConcreteScope,
) -> String {
    let children = first_occurrence_unique(scope.nodes.iter().filter_map(|node| node.child_scope));
    let mut proof = String::from("by\n      intro child member\n");
    if children.is_empty() {
        writeln!(
            proof,
            "      simp [{scope_name}, Mxx.IR.structuralChildren, Mxx.IR.NodePayload.childScope?] at member"
        )
        .unwrap();
        return proof.trim_end().to_owned();
    }
    writeln!(
        proof,
        "      have cases : {} := by simpa [{scope_name}, Mxx.IR.structuralChildren, Mxx.IR.NodePayload.childScope?] using member",
        alternatives(children.iter().map(|child| format!("child = {child}")))
    )
    .unwrap();
    writeln!(
        proof,
        "      rcases cases with {}",
        std::iter::repeat_n("rfl", children.len()).collect::<Vec<_>>().join(" | ")
    )
    .unwrap();
    for child in children {
        writeln!(
            proof,
            "      · exact ⟨⟨stage{stage_index}_scope{child}, stage{stage_index}_scope{child}Stored⟩, by decide⟩"
        )
        .unwrap();
    }
    proof.trim_end().to_owned()
}

fn render_artifact_ports_proof(
    ast: &ConcreteLinkedProgram,
    stage_index: usize,
    scope: &crate::linked::ConcreteScope,
    node_index: usize,
    node: &ConcreteNode,
    node_name: &str,
) -> Result<String, LeanEmissionError> {
    let ConcreteNodePayload::Input { artifact: Some(artifact), .. } = &node.kind else {
        return Ok(format!("by\n    simp [{node_name}]"));
    };
    let link_names = (0..ast.artifact_links.len())
        .map(|index| format!("artifactLink{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut port_links = Vec::with_capacity(node.outputs.len());
    for port in 0..node.outputs.len() {
        let matches = ast
            .artifact_links
            .iter()
            .enumerate()
            .filter(|(_, link)| {
                link.consumer_stage == stage_index &&
                    link.consumer.scope == scope.id &&
                    link.consumer.node.0 as usize == node_index &&
                    link.consumer.port.0 as usize == port &&
                    link.argument == port
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(LeanEmissionError::Encoding {
                message: format!(
                    "artifact input {node_name} port {port} has {} concrete links",
                    matches.len()
                ),
            });
        }
        port_links.push(matches[0]);
    }
    let input_index = *port_links.first().ok_or_else(|| LeanEmissionError::Encoding {
        message: format!("artifact input {node_name} has no outputs"),
    })?;
    if port_links.iter().any(|index| *index != input_index) {
        return Err(LeanEmissionError::Encoding {
            message: format!("artifact input {node_name} ports disagree on input index"),
        });
    }
    let input = format!(
        "{{ index := {input_index}, name := {}, confidentiality := {} }}",
        lean_string(&artifact.name),
        render_confidentiality(artifact.confidentiality)
    );
    let mut proof = format!(
        "by\n    intro input payload port portBound\n    have inputEq : input = {input} := by simpa [{node_name}] using payload.symm\n    subst input\n    change port < {} at portBound\n",
        node.outputs.len()
    );
    let port_cases = (0..node.outputs.len()).map(|port| format!("port = {port}"));
    writeln!(proof, "    have cases : {} := by omega", alternatives(port_cases)).unwrap();
    writeln!(
        proof,
        "    rcases cases with {}",
        std::iter::repeat_n("rfl", node.outputs.len()).collect::<Vec<_>>().join(" | ")
    )
    .unwrap();
    for link_index in port_links {
        proof.push_str("    · constructor\n");
        writeln!(proof, "      · exact ⟨artifactLink{link_index}, by rfl, by rfl, by rfl, by rfl⟩")
            .unwrap();
        proof.push_str(
            "      · intro foundIndex link stored consumerStage consumer argument\n        rcases Array.getElem?_eq_some_iff.mp stored with ⟨foundBound, foundEq⟩\n",
        );
        writeln!(proof, "        change foundIndex < {} at foundBound", ast.artifact_links.len())
            .unwrap();
        let cases = (0..ast.artifact_links.len()).map(|index| format!("foundIndex = {index}"));
        writeln!(proof, "        have cases : {} := by omega", alternatives(cases)).unwrap();
        writeln!(
            proof,
            "        rcases cases with {}",
            std::iter::repeat_n("rfl", ast.artifact_links.len()).collect::<Vec<_>>().join(" | ")
        )
        .unwrap();
        for found_index in 0..ast.artifact_links.len() {
            writeln!(
                proof,
                "        · simp [linkedProgramData, artifactLinks, {link_names}] at foundEq\n          subst link"
            )
            .unwrap();
            if found_index == link_index {
                proof.push_str("          rfl\n");
            } else {
                proof.push_str("          simp at consumerStage consumer argument\n");
            }
        }
    }
    Ok(proof.trim_end().to_owned())
}

fn render_stage_scope_id_lemma(
    source: &mut String,
    stage_index: usize,
    stage: &crate::linked::ConcreteLinkedStage,
) -> Result<(), LeanEmissionError> {
    for (index, scope) in stage.scopes.iter().enumerate() {
        if scope.id != index {
            return Err(LeanEmissionError::Encoding {
                message: format!(
                    "stage {stage_index} scope index {index} has non-canonical id {}",
                    scope.id
                ),
            });
        }
    }
    writeln!(
        source,
        "def stage{stage_index}ScopeStoredId : ∀ (index : Nat) (scope : Mxx.IR.Scope), stage{stage_index}.scopes[index]? = some scope → scope.id = index := by"
    )
    .unwrap();
    source.push_str(
        "  intro index scope stored\n  rcases Array.getElem?_eq_some_iff.mp stored with ⟨bound, stored⟩\n",
    );
    writeln!(source, "  change index < {} at bound", stage.scopes.len()).unwrap();
    if stage.scopes.is_empty() {
        source.push_str("  omega\n\n");
        return Ok(());
    }
    writeln!(
        source,
        "  have cases : {} := by omega",
        alternatives((0..stage.scopes.len()).map(|index| format!("index = {index}")))
    )
    .unwrap();
    writeln!(
        source,
        "  rcases cases with {}",
        std::iter::repeat_n("rfl", stage.scopes.len()).collect::<Vec<_>>().join(" | ")
    )
    .unwrap();
    for _ in &stage.scopes {
        writeln!(source, "  · simp [stage{stage_index}] at stored\n    subst scope\n    rfl")
            .unwrap();
    }
    source.push('\n');
    Ok(())
}

fn render_stage_named_outputs_proof(
    stage_index: usize,
    stage: &crate::linked::ConcreteLinkedStage,
) -> Result<String, LeanEmissionError> {
    let mut proof = String::from("by\n    intro output member\n");
    if stage.named_outputs.is_empty() {
        writeln!(proof, "    simp [stage{stage_index}] at member").unwrap();
        return Ok(proof.trim_end().to_owned());
    }
    let rendered_outputs = stage.named_outputs.iter().map(|output| {
        format!(
            "output = {{ name := {}, wire := {} }}",
            lean_string(&output.name),
            render_wire(&output.wire)
        )
    });
    writeln!(
        proof,
        "    have cases : {} := by simpa [stage{stage_index}] using member",
        alternatives(rendered_outputs)
    )
    .unwrap();
    writeln!(
        proof,
        "    rcases cases with {}",
        std::iter::repeat_n("rfl", stage.named_outputs.len()).collect::<Vec<_>>().join(" | ")
    )
    .unwrap();
    for output in &stage.named_outputs {
        let scope =
            stage.scopes.iter().find(|scope| scope.id == output.wire.scope).ok_or_else(|| {
                LeanEmissionError::Encoding {
                    message: format!("named output {} has missing scope", output.name),
                }
            })?;
        writeln!(
            proof,
            "    · exact ⟨{}, by rfl⟩",
            render_type(resolve_scope_wire_type(scope, &output.wire)?)?
        )
        .unwrap();
    }
    Ok(proof.trim_end().to_owned())
}

fn render_program_certificate(
    source: &mut String,
    ast: &ConcreteLinkedProgram,
) -> Result<(), LeanEmissionError> {
    let mut stage_leaf_names = Vec::with_capacity(ast.stages.len());
    let mut coverage_stage_leaf_names = Vec::with_capacity(ast.stages.len());
    for (stage_index, stage) in ast.stages.iter().enumerate() {
        let ranks = scope_ranks(stage)?;
        writeln!(source, "def stage{stage_index}ScopeRank : Mxx.IR.ScopeId → Nat").unwrap();
        for (scope, rank) in stage.scopes.iter().zip(ranks.iter()) {
            writeln!(source, "  | {} => {rank}", scope.id).unwrap();
        }
        source.push_str("  | _ => 0\n\n");
        render_stage_scope_id_lemma(source, stage_index, stage)?;

        let mut scope_leaf_names = Vec::with_capacity(stage.scopes.len());
        let mut coverage_scope_leaf_names = Vec::with_capacity(stage.scopes.len());
        for (scope_index, scope) in stage.scopes.iter().enumerate() {
            let scope_name = format!("stage{stage_index}_scope{}", scope.id);
            let scope_cert = format!("{scope_name}Cert");
            scope_leaf_names.push(scope_cert.clone());
            writeln!(
                source,
                "def {scope_cert} : Mxx.IR.StoredScopeCert stage{stage_index} stage{stage_index}ScopeRank {scope_index} where"
            )
            .unwrap();
            writeln!(source, "  scope := {scope_name}\n  stored := by rfl\n  valid := {{").unwrap();
            writeln!(source, "    slots := {}", render_scope_slots_proof(&scope_name, scope))
                .unwrap();
            writeln!(
                source,
                "    inputs := {}",
                render_scope_wires_proof(&scope_name, scope, &scope.inputs)?
            )
            .unwrap();
            writeln!(
                source,
                "    outputs := {}",
                render_scope_wires_proof(&scope_name, scope, &scope.outputs)?
            )
            .unwrap();
            writeln!(
                source,
                "    childrenDecrease := {}\n  }}",
                render_children_decrease_proof(stage_index, &scope_name, scope)
            )
            .unwrap();
            writeln!(source, "  nodes := {scope_name}Nodes\n").unwrap();

            let coverage_scope = format!("{scope_name}Coverage");
            coverage_scope_leaf_names.push(coverage_scope.clone());
            let mut coverage_node_names = Vec::with_capacity(scope.nodes.len());
            for node_index in 0..scope.nodes.len() {
                let node_name = format!("{scope_name}_node{node_index}");
                let coverage_node = format!("{node_name}Coverage");
                coverage_node_names.push(coverage_node.clone());
                writeln!(
                    source,
                    "def {coverage_node} : Mxx.IR.ArtifactNodeCoverageCert linkedProgramData {stage_index} stage{stage_index} {scope_index} {scope_name} {node_index} where"
                )
                .unwrap();
                writeln!(source, "  node := {node_name}\n  stored := by rfl").unwrap();
                writeln!(
                    source,
                    "  ports := {}\n",
                    render_artifact_ports_proof(
                        ast,
                        stage_index,
                        scope,
                        node_index,
                        &scope.nodes[node_index],
                        &node_name,
                    )?
                )
                .unwrap();
            }
            render_named_range(
                source,
                &format!("{scope_name}CoverageNodes"),
                &format!(
                    "Mxx.IR.ArtifactNodeRangeCert linkedProgramData {stage_index} stage{stage_index} {scope_index} {scope_name}"
                ),
                "Mxx.IR.ArtifactNodeRangeCert",
                &coverage_node_names,
            );
            writeln!(
                source,
                "def {coverage_scope} : Mxx.IR.ArtifactScopeCoverageCert linkedProgramData {stage_index} stage{stage_index} {scope_index} where\n  scope := {scope_name}\n  stored := by rfl\n  nodes := {scope_name}CoverageNodes\n"
            )
            .unwrap();
        }
        render_named_range(
            source,
            &format!("stage{stage_index}Scopes"),
            &format!("Mxx.IR.ScopeRangeCert stage{stage_index} stage{stage_index}ScopeRank"),
            "Mxx.IR.ScopeRangeCert",
            &scope_leaf_names,
        );
        let stage_cert = format!("stage{stage_index}Cert");
        stage_leaf_names.push(stage_cert.clone());
        writeln!(
            source,
            "def {stage_cert} : Mxx.IR.StoredStageCert linkedProgramData {stage_index} where\n  stage := stage{stage_index}\n  stored := by rfl\n  nonempty := by decide\n  rootStored := ⟨stage{stage_index}_scope{}, stage{stage_index}_scope{}Stored⟩\n  uniqueIds := by\n    intro first second firstScope secondScope firstStored secondStored different same\n    have firstId := stage{stage_index}ScopeStoredId first firstScope firstStored\n    have secondId := stage{stage_index}ScopeStoredId second secondScope secondStored\n    exact different (firstId.symm.trans (same.trans secondId))\n  namedOutputs := {}\n  rankOf := stage{stage_index}ScopeRank\n  scopes := stage{stage_index}Scopes\n",
            stage.root_scope,
            stage.root_scope,
            render_stage_named_outputs_proof(stage_index, stage)?
        )
        .unwrap();

        render_named_range(
            source,
            &format!("stage{stage_index}CoverageScopes"),
            &format!(
                "Mxx.IR.ArtifactScopeRangeCert linkedProgramData {stage_index} stage{stage_index}"
            ),
            "Mxx.IR.ArtifactScopeRangeCert",
            &coverage_scope_leaf_names,
        );
        let coverage_stage = format!("stage{stage_index}Coverage");
        coverage_stage_leaf_names.push(coverage_stage.clone());
        writeln!(
            source,
            "def {coverage_stage} : Mxx.IR.ArtifactStageCoverageCert linkedProgramData {stage_index} where\n  stage := stage{stage_index}\n  stored := by rfl\n  scopes := stage{stage_index}CoverageScopes\n"
        )
        .unwrap();
    }

    render_named_range(
        source,
        "linkedProgramStages",
        "Mxx.IR.StageRangeCert linkedProgramData",
        "Mxx.IR.StageRangeCert",
        &stage_leaf_names,
    );

    let mut link_leaf_names = Vec::with_capacity(ast.artifact_links.len());
    for link_index in 0..ast.artifact_links.len() {
        let link = &ast.artifact_links[link_index];
        let name = format!("artifactLink{link_index}Cert");
        link_leaf_names.push(name.clone());
        writeln!(
            source,
            "def {name} : Mxx.IR.StoredLinkCert linkedProgramData {link_index} where\n  link := artifactLink{link_index}\n  stored := by rfl\n  valid := {{\n    stored := by rfl\n    order := by decide\n    consumerStage := ⟨stage{}, by rfl⟩\n    producerStage := ⟨stage{}, by rfl⟩\n    consumerStored := ⟨stage{}, stage{}_scope{}, stage{}_scope{}_node{}, _, by rfl, by rfl, by rfl, by rfl, by rfl, by rfl, by rfl⟩\n    consumerTypeStored := ⟨stage{}, by rfl, by rfl⟩\n    producerTypeStored := ⟨stage{}, by rfl, by rfl⟩\n    typeCompatible := by simp [Mxx.IR.structuralTypeCompatible]\n    artifactName := by rfl\n    confidentiality := by rfl\n    argumentPort := by rfl\n  }}\n",
            link.consumer_stage,
            link.producer_stage,
            link.consumer_stage,
            link.consumer_stage,
            link.consumer.scope,
            link.consumer_stage,
            link.consumer.scope,
            link.consumer.node.0,
            link.consumer_stage,
            link.producer_stage,
        )
        .unwrap();
    }
    render_named_range(
        source,
        "linkedProgramLinks",
        "Mxx.IR.LinkRangeCert linkedProgramData",
        "Mxx.IR.LinkRangeCert",
        &link_leaf_names,
    );
    render_named_range(
        source,
        "linkedProgramCoverageStages",
        "Mxx.IR.ArtifactStageRangeCert linkedProgramData",
        "Mxx.IR.ArtifactStageRangeCert",
        &coverage_stage_leaf_names,
    );
    source.push_str(
        "def linkedProgramArtifactCoverage : Mxx.IR.ArtifactCoverageCert linkedProgramData :=\n  { stages := linkedProgramCoverageStages }\n\ndef linkedProgramCertificate : Mxx.IR.ProgramData.Certificate linkedProgramData := {\n  stages := linkedProgramStages\n  links := linkedProgramLinks\n  artifactCoverage := linkedProgramArtifactCoverage\n}\n\nnoncomputable def program : Mxx.IR.Program := {\n  data := linkedProgramData\n  valid := linkedProgramCertificate.sound\n}\n\n",
    );
    Ok(())
}

/// Return the proof constructor for payloads delegated to `evalPrimitiveNode`.
/// Inputs, samplers, family operations, and structural nodes have separate evaluator branches.
fn render_primitive_payload_proof(
    kind: &ConcreteNodePayload,
) -> Result<Option<String>, LeanEmissionError> {
    use ConcreteNodePayload as NodeKind;
    let proof = match kind {
        NodeKind::ConstantInt(value) => format!("Mxx.IR.PrimitiveNodePayload.constantInt {value}"),
        NodeKind::EvaluateInt(value) => {
            format!("Mxx.IR.PrimitiveNodePayload.evaluateInt {}", render_int(value))
        }
        NodeKind::ConstantBool(value) => {
            format!("Mxx.IR.PrimitiveNodePayload.constantBool {value}")
        }
        NodeKind::ConstantMatrix { matrix_type, value } => {
            if matches!(value, ConcreteMatrixLiteral::Gadget { small: true, .. }) {
                return Ok(None);
            }
            format!(
                "Mxx.IR.PrimitiveNodePayload.constantMatrix {} ({}) (by simp)",
                render_matrix_expr(matrix_type)?,
                render_constant_matrix(value)
            )
        }
        NodeKind::IntBinary(op) => format!(
            "Mxx.IR.PrimitiveNodePayload.intBinary {}",
            match op {
                crate::node::IntBinaryOp::Add => ".add",
                crate::node::IntBinaryOp::Subtract => ".subtract",
                crate::node::IntBinaryOp::Multiply => ".multiply",
                crate::node::IntBinaryOp::Divide => ".divide",
                crate::node::IntBinaryOp::Remainder => ".remainder",
            }
        ),
        NodeKind::IntCompare(op) => format!(
            "Mxx.IR.PrimitiveNodePayload.intCompare {}",
            match op {
                crate::node::IntCompareOp::Equal => ".equal",
                crate::node::IntCompareOp::Less => ".less",
                crate::node::IntCompareOp::LessEqual => ".lessEqual",
            }
        ),
        NodeKind::BitExtract { bit } => {
            format!("Mxx.IR.PrimitiveNodePayload.bitExtract {}", render_int(bit))
        }
        NodeKind::IntToReal => "Mxx.IR.PrimitiveNodePayload.intToReal".to_owned(),
        NodeKind::BoolToInt => "Mxx.IR.PrimitiveNodePayload.boolToInt".to_owned(),
        NodeKind::RealBinary(op) => {
            format!("Mxx.IR.PrimitiveNodePayload.realBinary {}", render_real_binary(*op))
        }
        NodeKind::RealSqrt => "Mxx.IR.PrimitiveNodePayload.realSqrt".to_owned(),
        NodeKind::MatrixBinary(op) => format!(
            "Mxx.IR.PrimitiveNodePayload.matrixBinary {}",
            match op {
                crate::node::MatrixBinaryOp::Add => ".add",
                crate::node::MatrixBinaryOp::Subtract => ".subtract",
                crate::node::MatrixBinaryOp::Multiply => ".multiply",
            }
        ),
        NodeKind::MatrixNegate => "Mxx.IR.PrimitiveNodePayload.matrixNegate".to_owned(),
        NodeKind::MatrixScale { scalar } => {
            format!("Mxx.IR.PrimitiveNodePayload.matrixScale {}", render_int(scalar))
        }
        NodeKind::Transpose => "Mxx.IR.PrimitiveNodePayload.transpose".to_owned(),
        NodeKind::Slice { rows, columns } => format!(
            "Mxx.IR.PrimitiveNodePayload.slice ({}) ({})",
            render_range(rows),
            render_range(columns)
        ),
        NodeKind::Concat { axis } => format!(
            "Mxx.IR.PrimitiveNodePayload.concat {}",
            match axis {
                crate::node::ConcatAxis::Rows => ".rows",
                crate::node::ConcatAxis::Columns => ".columns",
                crate::node::ConcatAxis::Diagonal => ".diagonal",
            }
        ),
        NodeKind::GadgetDecompose { base, small, digit_count } => {
            if *small {
                return Ok(None);
            }
            format!(
                "Mxx.IR.PrimitiveNodePayload.gadgetDecompose {} {}",
                render_int(base),
                render_int(digit_count)
            )
        }
        NodeKind::ExtractCoefficient { position, canonical_input_exclusive_upper } => format!(
            "Mxx.IR.PrimitiveNodePayload.extractCoefficient {} {}",
            render_int(position),
            render_optional_big(canonical_input_exclusive_upper)
        ),
        NodeKind::TrapdoorPublic => "Mxx.IR.PrimitiveNodePayload.trapdoorPublic".to_owned(),
        NodeKind::ApplyPreimage => "Mxx.IR.PrimitiveNodePayload.applyPreimage".to_owned(),
        NodeKind::MaterializePreimageExact => {
            "Mxx.IR.PrimitiveNodePayload.materializePreimageExact".to_owned()
        }
        _ => return Ok(None),
    };
    Ok(Some(proof))
}

fn render_scope_free_family_payload_proof(kind: &ConcreteNodePayload) -> Option<String> {
    let proof = match kind {
        ConcreteNodePayload::FamilyGetStatic { indices } => format!(
            "Mxx.IR.ScopeFreeFamilyPayload.getStatic #[{}]",
            indices.iter().map(render_index).collect::<Vec<_>>().join(", ")
        ),
        ConcreteNodePayload::FamilyGetDynamic { rank } => {
            format!("Mxx.IR.ScopeFreeFamilyPayload.getDynamic {rank}")
        }
        ConcreteNodePayload::FamilySelectAxis { axis } => {
            format!("Mxx.IR.ScopeFreeFamilyPayload.selectAxis {axis}")
        }
        ConcreteNodePayload::FamilyReindex { output_shape, map } => format!(
            "Mxx.IR.ScopeFreeFamilyPayload.reindex #[{}] {}",
            render_ints(output_shape),
            render_index_map(map)
        ),
        _ => return None,
    };
    Some(proof)
}

fn render_node_equation_scopes(
    ast: &ConcreteLinkedProgram,
) -> Result<Vec<(String, String)>, LeanEmissionError> {
    let mut rendered = Vec::new();
    for (stage_index, stage) in ast.stages.iter().enumerate() {
        for scope in &stage.scopes {
            let scope_name = format!("stage{stage_index}_scope{}", scope.id);
            let mut source = String::new();
            for (node_index, node) in scope.nodes.iter().enumerate() {
                let node_name = format!("{scope_name}_node{node_index}");
                match &node.kind {
                    ConcreteNodePayload::Input { artifact: Some(artifact), .. } => {
                        let artifact_name = format!("{node_name}Artifact");
                        let artifact_index =
                            artifact_index(ast, stage_index, scope.id, node_index)?;
                        writeln!(
                            source,
                            "abbrev {artifact_name} : Mxx.IR.ArtifactInput := {{ index := {artifact_index}, name := {}, confidentiality := {} }}\n",
                            lean_string(&artifact.name),
                            render_confidentiality(artifact.confidentiality)
                        )
                        .unwrap();
                        render_artifact_equation(
                            &mut source,
                            stage_index,
                            scope.id,
                            node_index,
                            &node_name,
                            &scope_name,
                            &artifact_name,
                        );
                    }
                    ConcreteNodePayload::Input { artifact: None, .. } => {
                        render_input_equation(
                            &mut source,
                            stage_index,
                            scope.id,
                            node_index,
                            &node_name,
                            &scope_name,
                        );
                    }
                    kind if matches!(
                        kind,
                        ConcreteNodePayload::UniformResidueSample { .. } |
                            ConcreteNodePayload::UniformIntervalSample { .. } |
                            ConcreteNodePayload::GaussianSample { .. } |
                            ConcreteNodePayload::HashSample { .. } |
                            ConcreteNodePayload::TrapdoorSample { .. } |
                            ConcreteNodePayload::PreimageSample { .. } |
                            ConcreteNodePayload::FamilyPreimageSample { .. } |
                            ConcreteNodePayload::GadgetTrapdoor { .. }
                    ) =>
                    {
                        render_sampler_equation(
                            &mut source,
                            stage_index,
                            scope.id,
                            node_index,
                            &node_name,
                            &scope_name,
                        );
                    }
                    ConcreteNodePayload::SubgraphCall(payload) => {
                        let child = child_scope(node, stage, scope.id)?;
                        let call_name = format!("{node_name}EquationCall");
                        writeln!(
                            source,
                            "abbrev {call_name} : Mxx.IR.SubgraphPayload := {}\n",
                            render_subgraph_payload(payload, child)
                        )
                        .unwrap();
                        render_subgraph_equation(
                            &mut source,
                            stage_index,
                            scope.id,
                            node_index,
                            &node_name,
                            &scope_name,
                            &call_name,
                        );
                    }
                    ConcreteNodePayload::SequentialLoop(payload) => {
                        let child = child_scope(node, stage, scope.id)?;
                        let loop_name = format!("{node_name}EquationLoop");
                        writeln!(
                            source,
                            "abbrev {loop_name} : Mxx.IR.LoopPayload := {}\n",
                            render_loop_payload(payload, child)
                        )
                        .unwrap();
                        render_loop_equation(
                            &mut source,
                            stage_index,
                            scope.id,
                            node_index,
                            &node_name,
                            &scope_name,
                            &loop_name,
                        );
                    }
                    ConcreteNodePayload::ParallelGrid(payload) => {
                        let child = child_scope(node, stage, scope.id)?;
                        let grid_name = format!("{node_name}EquationGrid");
                        writeln!(
                            source,
                            "abbrev {grid_name} : Mxx.IR.GridPayload := {}\n",
                            render_grid_payload(payload, child)
                        )
                        .unwrap();
                        render_grid_equation(
                            &mut source,
                            stage_index,
                            scope.id,
                            node_index,
                            &node_name,
                            &scope_name,
                            &grid_name,
                        );
                    }
                    kind => {
                        let Some(payload_proof) = render_primitive_payload_proof(kind)? else {
                            continue;
                        };
                        render_primitive_equation(
                            &mut source,
                            stage_index,
                            scope.id,
                            node_index,
                            &node_name,
                            &scope_name,
                            &payload_proof,
                        );
                    }
                }
            }
            // A suffix certificate refers to the equations of every later node, so emit it only
            // after those declarations are in scope.  The concrete loop payload supplies the
            // child scope used in the selected iteration occurrence frame.
            for (node_index, node) in scope.nodes.iter().enumerate() {
                if matches!(node.kind, ConcreteNodePayload::SequentialLoop(_)) {
                    render_loop_suffix_avoidance(
                        &mut source,
                        stage_index,
                        scope,
                        node_index,
                        child_scope(node, stage, scope.id)?,
                        &format!("{scope_name}_node{node_index}"),
                        &scope_name,
                    )?;
                }
            }
            for target in 1..scope.nodes.len() {
                if let Some(prefix) =
                    render_flat_scope_prefix_callback(stage_index, scope, target, false)?
                {
                    render_prefix_steps_theorem(
                        &mut source,
                        stage_index,
                        scope.id,
                        target,
                        &scope_name,
                        &prefix,
                    );
                }
                if let Some(prefix) = render_scope_free_prefix_callback(stage_index, scope, target)?
                {
                    render_scope_free_prefix_steps_theorem(
                        &mut source,
                        stage_index,
                        scope.id,
                        target,
                        &scope_name,
                        &prefix,
                    );
                }
            }
            if !source.is_empty() {
                rendered.push((scope_name, source));
            }
        }
    }
    Ok(rendered)
}

/// Emit an execution-extraction hook for a root-scope gadget node.  The generated theorem starts
/// from the whole-program evaluator result, then uses the generated prefix and node equations to
/// recover the exact primitive call.  Cryptographic certificate interpretation belongs to the
/// runtime backend, not to this backend-agnostic IR module.
fn render_gadget_execution_hook(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    base: &crate::linked::ConcreteStructuralIntExpr,
    small: bool,
    digit_count: &crate::linked::ConcreteStructuralIntExpr,
) {
    let base = render_int(base);
    let digits = render_int(digit_count);
    let small = if small { "true" } else { "false" };
    let hook_start = source.len();
    writeln!(
        source,
        "theorem {node_name}GadgetCertificate {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (finalTrace : Mxx.IR.Trace backend)\n  (success : Mxx.IR.eval backend program env = .ok finalTrace) :\n  ∃ targetType outputType target output,\n    Nonempty (Mxx.IR.GadgetDecomposeExecution backend {{}} {stage_index} {scope} {node_index}\n      ({base}) {small} ({digits}) targetType outputType target output) := by\n  obtain ⟨tracePrefix, values, fuel, finalResult, nodeSuccess, fuelEq⟩ :=\n    {node_name}FromPublicEval env finalTrace success\n  have fuelPositive : fuel ≠ 0 := by\n    rw [fuelEq]\n    have fuelBound : {node_index} < Mxx.IR.evaluationFuel linkedProgramData := by decide\n    omega\n  obtain ⟨argumentValues, nodeResult, nextResult, _, primitiveSuccess, _, _, _⟩ :=\n    {node_name}Equation env {{}} {{ stages := tracePrefix }} #[] #[] values fuel finalResult\n      fuelPositive nodeSuccess\n  obtain ⟨targetType, outputType, target, output, _, _, _, execution⟩ :=\n    Mxx.IR.evalPrimitiveNode_gadgetDecompose_success {{}} {stage_index} {scope} {node_index}\n      ({base}) {small} ({digits}) argumentValues {node_name}.outputs nodeResult primitiveSuccess\n  exact ⟨targetType, outputType, target, output, execution⟩\n\n-- The hook recovers the evaluated digit expression: {digits}; node index: {stage_index}:{scope}:{node_index}.\n",
    )
    .unwrap();
    let hook = source.split_off(hook_start);
    source.push_str(&hook.replace("GadgetCertificate", "GadgetExecution"));
}

fn render_artifact_trace_hook(
    source: &mut String,
    stage_index: usize,
    scope_id: usize,
    target: usize,
    node_name: &str,
    prefix: &str,
) {
    let artifact_name = format!("{node_name}Artifact");
    let prefix_body = if target == 0 {
        "    intro tracePrefix limit index values fuel finalResult limitEq indexBound fuelPositive targetSuccess\n    omega\n"
            .to_owned()
    } else {
        let nested_prefix =
            prefix.lines().map(|line| format!("  {line}")).collect::<Vec<_>>().join("\n");
        format!(
            "    intro tracePrefix limit index values fuel finalResult limitEq indexBound fuelPositive targetSuccess\n    let trace : Mxx.IR.Trace backend := {{ stages := tracePrefix }}\n  {nested_prefix}\n    simpa [trace] using\n      (prefixStep limit index values fuel finalResult limitEq indexBound fuelPositive targetSuccess)\n"
        )
    };
    writeln!(
        source,
        "theorem {node_name}ArtifactValueFromPublicEval {{backend : Mxx.IR.SemanticBackend}}\n  \
(env : Mxx.IR.EvalEnv backend linkedProgramData)\n  \
(finalTrace : Mxx.IR.Trace backend)\n  \
(success : Mxx.IR.eval backend program env = .ok finalTrace) :\n  \
∃ link producerTrace producerScope value,\n    \
linkedProgramData.artifactLinks[{artifact_name}.index]? = some link ∧\n    \
finalTrace.stages[link.producerStage]? = some producerTrace ∧\n    \
producerTrace.scopes.find? (fun item ↦ item.scope = link.producer.scope) = some producerScope ∧\n    \
Mxx.IR.lookup producerScope.values link.producer = some value ∧\n    \
Mxx.IR.traceValueAt finalTrace\n      \
(Mxx.IR.occurrenceOf {stage_index} #[] {{ scope := {scope_id}, node := {target}, port := 0 }}) =\n        \
some value := by\n  \
have prefixSteps : ∀ (tracePrefix : Array (Mxx.IR.StageTrace backend))\n    \
(limit index : Nat) (values : Array (Mxx.IR.Binding backend))\n    \
(fuel : Nat) (finalResult : Mxx.IR.ScopeResult backend),\n    \
limit = {target} → index < limit → fuel ≠ 0 →\n    \
Mxx.IR.evalScope linkedProgramData env {{}} {{ stages := tracePrefix }} {stage_index}\n      \
stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id} (by rfl) (by rfl)\n      \
#[] #[] index values fuel = .ok finalResult →\n    \
Mxx.IR.FlatScopeStep linkedProgramData env {{}} {{ stages := tracePrefix }} {stage_index}\n      \
stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id} (by rfl) (by rfl)\n      \
#[] #[] index values fuel finalResult := by\n{prefix_body}  \
exact Mxx.IR.eval_success_root_artifact_input_at program env finalTrace {stage_index}\n    \
(by decide) success stage{stage_index} (by rfl) stage{stage_index}_scope{scope_id} (by rfl)\n    \
{target} (by decide) prefixSteps {node_name} (by rfl) {artifact_name} (by rfl)\n"
    )
    .unwrap();
}

fn render_stage_root_equations(
    ast: &ConcreteLinkedProgram,
) -> Result<Vec<(String, String)>, LeanEmissionError> {
    let mut rendered = Vec::with_capacity(ast.stages.len());
    for (stage_index, stage_data) in ast.stages.iter().enumerate() {
        let theorem_name = format!("stage{stage_index}RootSuccess");
        let mut source = format!(
            "theorem {theorem_name} {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (finalTrace : Mxx.IR.Trace backend)\n  (success : Mxx.IR.eval backend program env = .ok finalTrace) :\n  ∃ tracePrefix : Array (Mxx.IR.StageTrace backend), ∃ stage,\n    ∃ stageStored : linkedProgramData.stages[{stage_index}]? = some stage, ∃ stageTrace,\n    ∃ root, ∃ rootStored : Mxx.IR.scopeAt stage stage.root = some root, ∃ result,\n    Mxx.IR.evalStages linkedProgramData env {stage_index} {{ stages := tracePrefix }} = .ok finalTrace ∧\n    Mxx.IR.evalStage linkedProgramData env {{ stages := tracePrefix }} {stage_index} stage stageStored = .ok stageTrace ∧\n    Mxx.IR.evalScope linkedProgramData env {{}} {{ stages := tracePrefix }} {stage_index} stage stage.root root\n      stageStored rootStored #[] #[] 0 #[] (Mxx.IR.evaluationFuel linkedProgramData) = .ok result ∧\n    stageTrace = {{ stage := {stage_index}, scopes := result.scopes }} := by\n  exact Mxx.IR.generatedStageRootSuccessAt linkedProgramData env finalTrace {stage_index} (by decide)\n    (Mxx.IR.eval_success_stages program env finalTrace success)\n"
        );
        if let Some(scope) =
            stage_data.scopes.iter().find(|scope| scope.id == stage_data.root_scope)
        {
            if let Some(node) = scope.nodes.first() {
                if matches!(node.kind, ConcreteNodePayload::Input { artifact: Some(_), .. }) {
                    let node_name = format!("stage{stage_index}_scope{}_node0", scope.id);
                    render_artifact_trace_hook(
                        &mut source,
                        stage_index,
                        scope.id,
                        0,
                        &node_name,
                        "",
                    );
                }
            }
            for target in 1..scope.nodes.len() {
                let Some(prefix) =
                    render_flat_scope_prefix_callback(stage_index, scope, target, true)?
                else {
                    continue;
                };
                let node_name = format!("stage{stage_index}_scope{}_node{target}", scope.id);
                writeln!(
                    source,
                    "theorem {node_name}FromPublicEval {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (finalTrace : Mxx.IR.Trace backend)\n  (success : Mxx.IR.eval backend program env = .ok finalTrace) :\n  ∃ tracePrefix values fuel result,\n    Mxx.IR.evalScope linkedProgramData env {{}} {{ stages := tracePrefix }} {stage_index}\n      stage{stage_index} stage{stage_index}_scope{scope_id} (by rfl) (by rfl) #[] #[] {target}\n      values fuel = .ok result ∧\n    fuel = Mxx.IR.evaluationFuel linkedProgramData - {target} := by\n  obtain ⟨tracePrefix, stage, stageStored, stageTrace, root, rootStored, rootResult, _, _, rootSuccess, _⟩ :=\n    stage{stage_index}RootSuccess env finalTrace success\n  cases stageStored\n  cases rootStored\n  let trace : Mxx.IR.Trace backend := {{ stages := tracePrefix }}\n  {prefix}\n  obtain ⟨values, fuel, result, targetSuccess, fuelEq⟩ :=\n    Mxx.IR.generatedFlatScopePrefixAt linkedProgramData env {{}} trace {stage_index}\n      stage{stage_index}_scope{scope_id} {scope_id} (by rfl) (by rfl) #[]\n      (Mxx.IR.evaluationFuel linkedProgramData) {target} (by decide) prefixStep rootResult rootSuccess\n  exact ⟨tracePrefix, values, fuel, result, targetSuccess, fuelEq⟩\n",
                    scope_id = scope.id,
                )
                .unwrap();
                if let Some(payload_proof) =
                    render_primitive_payload_proof(&scope.nodes[target].kind)?
                {
                    let scope_id = scope.id;
                    let nested_prefix = prefix
                        .lines()
                        .map(|line| format!("  {line}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    writeln!(
                        source,
                        "theorem {node_name}ReachedPrimitiveRunFromPublicEval\n  \
{{backend : Mxx.IR.SemanticBackend}}\n  \
(env : Mxx.IR.EvalEnv backend linkedProgramData)\n  \
(finalTrace : Mxx.IR.Trace backend)\n  \
(success : Mxx.IR.eval backend program env = .ok finalTrace) :\n  \
Nonempty (Mxx.IR.ReachedPrimitiveRun finalTrace {{}} {stage_index} {scope_id} {target} #[]\n    \
{node_name}.payload {node_name} 0) := by\n  \
have prefixSteps : ∀ (tracePrefix : Array (Mxx.IR.StageTrace backend))\n    \
(limit index : Nat) (values : Array (Mxx.IR.Binding backend))\n    \
(fuel : Nat) (finalResult : Mxx.IR.ScopeResult backend),\n    \
limit = {target} → index < limit → fuel ≠ 0 →\n    \
Mxx.IR.evalScope linkedProgramData env {{}} {{ stages := tracePrefix }} {stage_index}\n      \
stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id} (by rfl) (by rfl)\n      \
#[] #[] index values fuel = .ok finalResult →\n    \
Mxx.IR.FlatScopeStep linkedProgramData env {{}} {{ stages := tracePrefix }} {stage_index}\n      \
stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id} (by rfl) (by rfl)\n      \
#[] #[] index values fuel finalResult := by\n    \
intro tracePrefix limit index values fuel finalResult limitEq indexBound fuelPositive targetSuccess\n    \
let trace : Mxx.IR.Trace backend := {{ stages := tracePrefix }}\n  \
{nested_prefix}\n    \
simpa [trace] using\n      \
(prefixStep limit index values fuel finalResult limitEq indexBound fuelPositive targetSuccess)\n  \
exact Mxx.IR.reachedRootPrimitiveRun program env finalTrace {stage_index} (by decide) success\n    \
stage{stage_index} (by rfl) stage{stage_index}_scope{scope_id} (by rfl) {target}\n    \
(by decide) prefixSteps {node_name} (by rfl) {node_name}.payload (by rfl)\n    \
({payload_proof}) 0 (by decide)\n",
                    )
                    .unwrap();
                }
                if matches!(
                    scope.nodes[target].kind,
                    ConcreteNodePayload::Input { artifact: Some(_), .. }
                ) {
                    render_artifact_trace_hook(
                        &mut source,
                        stage_index,
                        scope.id,
                        target,
                        &node_name,
                        &prefix,
                    );
                }
                if let ConcreteNodePayload::GadgetDecompose { base, small, digit_count } =
                    &scope.nodes[target].kind
                {
                    if !*small {
                        render_gadget_execution_hook(
                            &mut source,
                            stage_index,
                            scope.id,
                            target,
                            &node_name,
                            base,
                            *small,
                            digit_count,
                        );
                    }
                }
            }
        }
        if let Some(scope) = ast.stages[stage_index]
            .scopes
            .iter()
            .find(|scope| scope.id == ast.stages[stage_index].root_scope)
        {
            let scope_id = scope.id;
            source = source.replace(
                &format!("stage{stage_index} stage{stage_index}_scope{scope_id}"),
                &format!("stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id}"),
            );
            source = source.replace(
                &format!("stage{stage_index}_scope{scope_id} {scope_id} (by rfl)"),
                &format!(
                    "stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id} (by rfl)"
                ),
            );
            source = source.replace(
                &format!("stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id} (by rfl) (by rfl) #[]\n      (Mxx.IR.evaluationFuel"),
                &format!("stage{stage_index} {scope_id} stage{stage_index}_scope{scope_id} (by rfl) (by rfl) #[] #[] #[]\n      (Mxx.IR.evaluationFuel"),
            );
        }
        rendered.push((theorem_name, source));
    }
    Ok(rendered)
}

/// Generate the branch callback consumed by `generatedFlatScopePrefixAt`.  The callback is
/// finite because the linked scope is concrete; each preceding node is discharged by the
/// equation emitted for that exact node.  Unsupported node forms are left to the existing
/// equation renderer rather than being silently approximated.
fn render_flat_scope_prefix_callback(
    stage_index: usize,
    scope: &crate::linked::ConcreteScope,
    target: usize,
    root_context: bool,
) -> Result<Option<String>, LeanEmissionError> {
    for node in &scope.nodes[..target] {
        let structural_or_sampled = matches!(
            &node.kind,
            ConcreteNodePayload::Input { .. } |
                ConcreteNodePayload::SubgraphCall(_) |
                ConcreteNodePayload::SequentialLoop(_) |
                ConcreteNodePayload::ParallelGrid(_) |
                ConcreteNodePayload::UniformResidueSample { .. } |
                ConcreteNodePayload::UniformIntervalSample { .. } |
                ConcreteNodePayload::GaussianSample { .. } |
                ConcreteNodePayload::HashSample { .. } |
                ConcreteNodePayload::TrapdoorSample { .. } |
                ConcreteNodePayload::PreimageSample { .. } |
                ConcreteNodePayload::FamilyPreimageSample { .. } |
                ConcreteNodePayload::GadgetTrapdoor { .. }
        );
        if !structural_or_sampled && render_primitive_payload_proof(&node.kind)?.is_none() {
            return Ok(None);
        }
    }
    let scope_name = format!("stage{stage_index}_scope{}", scope.id);
    let cases = (0..target).map(|index| format!("index = {index}")).collect::<Vec<_>>().join(" ∨ ");
    let patterns = std::iter::repeat_n("rfl", target).collect::<Vec<_>>().join(" | ");
    let mut branches = String::new();
    for index in 0..target {
        let node_name = format!("{scope_name}_node{index}");
        let (result_pattern, result_value) = match &scope.nodes[index].kind {
            ConcreteNodePayload::Input { artifact: Some(_), .. } => (
                "⟨_, _, _, _, value, nextResult, _, _, _, _, _, typesMatch, nextSuccess, finalStored⟩",
                "Mxx.IR.NodeResult.ofValues #[value]",
            ),
            ConcreteNodePayload::Input { artifact: None, .. } => (
                "⟨_, nodeResult, nextResult, _, _, typesMatch, nextSuccess, finalStored⟩",
                "nodeResult",
            ),
            ConcreteNodePayload::UniformResidueSample { .. } |
            ConcreteNodePayload::UniformIntervalSample { .. } |
            ConcreteNodePayload::GaussianSample { .. } |
            ConcreteNodePayload::HashSample { .. } |
            ConcreteNodePayload::TrapdoorSample { .. } |
            ConcreteNodePayload::PreimageSample { .. } |
            ConcreteNodePayload::FamilyPreimageSample { .. } |
            ConcreteNodePayload::GadgetTrapdoor { .. } => (
                "⟨_, sampled, nextResult, _, _, typesMatch, nextSuccess, finalStored⟩",
                "Mxx.IR.NodeResult.ofValues sampled",
            ),
            ConcreteNodePayload::SubgraphCall(_) => (
                "⟨_, _, _, childInputs, childResult, childOutputs, nextResult, _, _, _, _, typesMatch, nextSuccess, finalStored⟩",
                "{ values := childOutputs, scopes := childResult.scopes }",
            ),
            ConcreteNodePayload::ParallelGrid(_) => (
                "⟨_, _, _, _, _, laneResults, packed, nextResult, _, _, _, _, _, typesMatch, nextSuccess, finalStored⟩",
                "{ values := packed, scopes := laneResults.foldl (fun result item => result ++ item.2) #[] }",
            ),
            ConcreteNodePayload::SequentialLoop(_) => (
                "⟨_, _, _, loopResult, nextResult, _, _, typesMatch, nextSuccess, finalStored⟩",
                "{ values := loopResult.values, scopes := loopResult.scopes }",
            ),
            primitive if render_primitive_payload_proof(primitive)?.is_some() => (
                "⟨_, nodeResult, nextResult, _, _, typesMatch, nextSuccess, finalStored⟩",
                "nodeResult",
            ),
            _ => return Ok(None),
        };
        writeln!(
            branches,
            "    · obtain {result_pattern} :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      refine ⟨{node_name}, {result_value}, nextResult, by rfl, typesMatch, nextSuccess, ?_⟩\n      simpa [Mxx.IR.NodeResult.ofValues] using finalStored"
        )
        .unwrap();
    }
    let callback = format!(
        "have prefixStep : ∀ (index : Nat) (values : Array (Mxx.IR.Binding backend))\n      (fuel : Nat) (finalResult : Mxx.IR.ScopeResult backend),\n      index < {scope_name}.nodes.size → fuel ≠ 0 →\n      Mxx.IR.evalScope linkedProgramData env structural trace {stage_index}\n        stage{stage_index} {scope_name} (by rfl) (by rfl) inputs path index values fuel = .ok finalResult →\n      Mxx.IR.FlatScopeStep linkedProgramData env structural trace {stage_index}\n        stage{stage_index} {scope_name} (by rfl) (by rfl) inputs path index values fuel finalResult := by\n    intro index values fuel finalResult indexBound fuelPositive success\n    have indexCases : {cases} := by omega\n    rcases indexCases with {patterns}\n{branches}"
        )
        .replace("∀ (index : Nat)", "∀ (limit index : Nat)")
        .replace(
            &format!("index < {scope_name}.nodes.size → fuel"),
            &format!("limit = {target} → index < limit → fuel"),
        )
        .replace(
            "intro index values fuel finalResult indexBound fuelPositive success",
            "intro limit index values fuel finalResult limitEq indexBound fuelPositive success\n    subst limit",
        );
    let callback = callback.replace(
        &format!("stage{stage_index} {scope_name}"),
        &format!("stage{stage_index} {} {scope_name}", scope.id),
    );
    if root_context {
        Ok(Some(callback.replace("structural", "{}").replace(" inputs path ", " #[] #[] ")))
    } else {
        Ok(Some(callback))
    }
}

/// Generate the stronger prefix callback used when every preceding node contributes no nested
/// scopes.  This deliberately accepts only ordinary inputs and evaluator-supported primitives;
/// structural nodes, samplers, and artifact inputs remain explicit omissions until their own
/// equations provide an equally direct empty-scope witness.
fn render_scope_free_prefix_callback(
    stage_index: usize,
    scope: &crate::linked::ConcreteScope,
    target: usize,
) -> Result<Option<String>, LeanEmissionError> {
    for node in &scope.nodes[..target] {
        let supported = matches!(node.kind, ConcreteNodePayload::Input { artifact: None, .. }) ||
            render_primitive_payload_proof(&node.kind)?.is_some() ||
            render_scope_free_family_payload_proof(&node.kind).is_some();
        if !supported {
            return Ok(None);
        }
    }
    let scope_name = format!("stage{stage_index}_scope{}", scope.id);
    let cases = (0..target).map(|index| format!("index = {index}")).collect::<Vec<_>>().join(" ∨ ");
    let patterns = std::iter::repeat_n("rfl", target).collect::<Vec<_>>().join(" | ");
    let mut branches = String::new();
    for index in 0..target {
        let node_name = format!("{scope_name}_node{index}");
        match &scope.nodes[index].kind {
            ConcreteNodePayload::Input { artifact: None, .. } => {
                writeln!(
                    branches,
                    "    · obtain ⟨_, nodeResult, nextResult, _, resultStored, typesMatch, nextSuccess, finalStored⟩ :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      have resultEmpty : nodeResult.scopes = #[] := by\n        rcases resultStored with ⟨_, _, rfl⟩ | ⟨_, _, _, _, rfl⟩ <;> rfl\n      exact ⟨{node_name}, nodeResult, nextResult, by rfl, resultEmpty, typesMatch, nextSuccess, finalStored⟩"
                )
                .unwrap();
            }
            primitive if render_primitive_payload_proof(primitive)?.is_some() => {
                writeln!(
                    branches,
                    "    · obtain ⟨arguments, nodeResult, nextResult, _, primitiveSuccess, typesMatch, nextSuccess, finalStored⟩ :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      have resultEmpty : nodeResult.scopes = #[] :=\n        Mxx.IR.evalPrimitiveNode_success_scopes_empty structural {stage_index} {} {index}\n          {node_name}.payload arguments {node_name}.outputs nodeResult primitiveSuccess\n      exact ⟨{node_name}, nodeResult, nextResult, by rfl, resultEmpty, typesMatch, nextSuccess, finalStored⟩",
                    scope.id
                )
                .unwrap();
            }
            family => {
                let Some(family_proof) = render_scope_free_family_payload_proof(family) else {
                    return Ok(None);
                };
                writeln!(
                    branches,
                    "    · exact Mxx.IR.evalScope_success_scope_free_family_step linkedProgramData env
        structural trace {stage_index} stage{stage_index} {} {scope_name} (by rfl) (by rfl)
        inputs path {index} values fuel finalResult fuelPositive (by decide) {node_name} (by rfl)
        {node_name}.payload (by rfl) ({family_proof}) success",
                    scope.id
                )
                .unwrap();
            }
        }
    }
    Ok(Some(format!(
        "have prefixStep : ∀ (limit index : Nat) (values : Array (Mxx.IR.Binding backend))\n      (fuel : Nat) (finalResult : Mxx.IR.ScopeResult backend),\n      limit = {target} → index < limit → fuel ≠ 0 →\n      Mxx.IR.evalScope linkedProgramData env structural trace {stage_index}\n        stage{stage_index} {} {scope_name} (by rfl) (by rfl) inputs path index values fuel = .ok finalResult →\n      Mxx.IR.ScopeFreeStep linkedProgramData env structural trace {stage_index}\n        stage{stage_index} {} {scope_name} (by rfl) (by rfl) inputs path index values fuel finalResult := by\n    intro limit index values fuel finalResult limitEq indexBound fuelPositive success\n    subst limit\n    have indexCases : {cases} := by omega\n    rcases indexCases with {patterns}\n{branches}",
        scope.id, scope.id
    )))
}

/// Expose the finite prefix callback for any concrete scope. This theorem contains only stored
/// syntax and evaluator-step inversion; callers still obtain the dynamic child execution from
/// the public evaluator.
fn render_prefix_steps_theorem(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    target: usize,
    scope_name: &str,
    prefix: &str,
) {
    let node_name = format!("{scope_name}_node{target}");
    writeln!(
        source,
        "theorem {node_name}PrefixSteps {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath) :\n  ∀ (limit index : Nat) (values : Array (Mxx.IR.Binding backend))\n    (fuel : Nat) (finalResult : Mxx.IR.ScopeResult backend),\n    limit = {target} → index < limit → fuel ≠ 0 →\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index}\n      stage{stage_index} {scope} {scope_name} (by rfl) (by rfl)\n      inputs path index values fuel = .ok finalResult →\n    Mxx.IR.FlatScopeStep linkedProgramData env structural trace {stage_index}\n      stage{stage_index} {scope} {scope_name} (by rfl) (by rfl)\n      inputs path index values fuel finalResult := by\n  {prefix}\n  exact prefixStep\n"
    )
    .unwrap();
}

fn render_scope_free_prefix_steps_theorem(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    target: usize,
    scope_name: &str,
    prefix: &str,
) {
    let node_name = format!("{scope_name}_node{target}");
    writeln!(
        source,
        "theorem {node_name}ScopeFreePrefixSteps {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath) :\n  ∀ (limit index : Nat) (values : Array (Mxx.IR.Binding backend))\n    (fuel : Nat) (finalResult : Mxx.IR.ScopeResult backend),\n    limit = {target} → index < limit → fuel ≠ 0 →\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index}\n      stage{stage_index} {scope} {scope_name} (by rfl) (by rfl)\n      inputs path index values fuel = .ok finalResult →\n    Mxx.IR.ScopeFreeStep linkedProgramData env structural trace {stage_index}\n      stage{stage_index} {scope} {scope_name} (by rfl) (by rfl)\n      inputs path index values fuel finalResult := by\n  {prefix}\n  exact prefixStep\n"
    )
    .unwrap();
}

fn render_loop_suffix_avoidance(
    source: &mut String,
    stage_index: usize,
    scope: &crate::linked::ConcreteScope,
    loop_index: usize,
    loop_child: usize,
    loop_name: &str,
    scope_name: &str,
) -> Result<(), LeanEmissionError> {
    let start = loop_index + 1;
    let node_count = scope.nodes.len();
    if start >= scope.nodes.len() {
        return Ok(());
    }
    for node in &scope.nodes[start..] {
        let supported = matches!(
            node.kind,
            ConcreteNodePayload::Input { .. } |
                ConcreteNodePayload::UniformResidueSample { .. } |
                ConcreteNodePayload::UniformIntervalSample { .. } |
                ConcreteNodePayload::GaussianSample { .. } |
                ConcreteNodePayload::HashSample { .. } |
                ConcreteNodePayload::TrapdoorSample { .. } |
                ConcreteNodePayload::PreimageSample { .. } |
                ConcreteNodePayload::FamilyPreimageSample { .. } |
                ConcreteNodePayload::GadgetTrapdoor { .. } |
                ConcreteNodePayload::ParallelGrid(_) |
                ConcreteNodePayload::SequentialLoop(_)
        ) || render_primitive_payload_proof(&node.kind)?.is_some() ||
            render_scope_free_family_payload_proof(&node.kind).is_some();
        if !supported {
            return Ok(());
        }
    }
    let cases = (start..scope.nodes.len())
        .map(|index| format!("index = {index}"))
        .collect::<Vec<_>>()
        .join(" ∨ ");
    let cases = format!("{cases} ∨ {scope_name}.nodes.size ≠ {node_count}");
    let patterns = format!(
        "{} | sizeMismatch",
        std::iter::repeat_n("rfl", scope.nodes.len() - start).collect::<Vec<_>>().join(" | ")
    );
    let mut branches = String::new();
    for index in start..scope.nodes.len() {
        let node_name = format!("{scope_name}_node{index}");
        match &scope.nodes[index].kind {
            ConcreteNodePayload::Input { artifact: None, .. } => {
                writeln!(branches, "    · obtain ⟨_, nodeResult, nextResult, _, resultStored, typesMatch, nextSuccess, finalStored⟩ :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      have resultEmpty : nodeResult.scopes = #[] := by\n        rcases resultStored with ⟨_, _, rfl⟩ | ⟨_, _, _, _, rfl⟩ <;> rfl\n      exact Mxx.IR.ScopeFreeStep.avoiding (⟨{node_name}, nodeResult, nextResult, by rfl, resultEmpty, typesMatch, nextSuccess, finalStored⟩ : Mxx.IR.ScopeFreeStep _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)").unwrap();
            }
            ConcreteNodePayload::Input { artifact: Some(_), .. } => {
                writeln!(branches, "    · obtain ⟨_, _, _, _, value, nextResult, _, _, _, _, _, typesMatch, nextSuccess, finalStored⟩ :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      exact Mxx.IR.ScopeFreeStep.avoiding (⟨{node_name}, Mxx.IR.NodeResult.ofValues #[value], nextResult, by rfl, rfl, typesMatch, nextSuccess, by simpa [Mxx.IR.NodeResult.ofValues] using finalStored⟩ : Mxx.IR.ScopeFreeStep _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)").unwrap();
            }
            ConcreteNodePayload::UniformResidueSample { .. } |
            ConcreteNodePayload::UniformIntervalSample { .. } |
            ConcreteNodePayload::GaussianSample { .. } |
            ConcreteNodePayload::HashSample { .. } |
            ConcreteNodePayload::TrapdoorSample { .. } |
            ConcreteNodePayload::PreimageSample { .. } |
            ConcreteNodePayload::FamilyPreimageSample { .. } |
            ConcreteNodePayload::GadgetTrapdoor { .. } => {
                writeln!(branches, "    · obtain ⟨_, sampled, nextResult, _, _, typesMatch, nextSuccess, finalStored⟩ :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      exact Mxx.IR.ScopeFreeStep.avoiding (⟨{node_name}, Mxx.IR.NodeResult.ofValues sampled, nextResult, by rfl, rfl, typesMatch, nextSuccess, by simpa [Mxx.IR.NodeResult.ofValues] using finalStored⟩ : Mxx.IR.ScopeFreeStep _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)").unwrap();
            }
            ConcreteNodePayload::ParallelGrid(_) => {
                writeln!(
                    branches,
                    "    · have nodeEquation :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      exact nodeEquation.avoidingScopeStep selectedFrame selectedPath selectedUnder\n        (by\n          intro lane equality\n          have scopeEq := congrArg Mxx.IR.OccurrenceFrame.scope equality\n          exact (by decide : {scope_name}.id ≠ {loop_child}) scopeEq) (by rfl)"
                )
                .unwrap();
            }
            ConcreteNodePayload::SequentialLoop(_) => {
                writeln!(
                    branches,
                    "    · have nodeEquation :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      exact nodeEquation.avoidingScopeStep selectedFrame selectedPath selectedUnder\n        (by\n          intro childId iteration equality\n          have ownerEq := congrArg Mxx.IR.OccurrenceFrame.owner equality\n          exact (by decide : {index} ≠ {loop_index}) ownerEq) (by rfl)"
                )
                .unwrap();
            }
            primitive if render_primitive_payload_proof(primitive)?.is_some() => {
                writeln!(
                    branches,
                    "    · obtain ⟨arguments, nodeResult, nextResult, _, primitiveSuccess, typesMatch, nextSuccess, finalStored⟩ :=\n        {node_name}Equation env structural trace inputs path values fuel finalResult fuelPositive success\n      have resultEmpty : nodeResult.scopes = #[] :=\n        Mxx.IR.evalPrimitiveNode_success_scopes_empty structural {stage_index} {} {index}\n          {node_name}.payload arguments {node_name}.outputs nodeResult primitiveSuccess\n      exact ⟨{node_name}, nodeResult, nextResult, by rfl,\n        (by simp [resultEmpty]), typesMatch, nextSuccess, finalStored⟩",
                    scope.id
                )
                .unwrap();
            }
            family => {
                let Some(family_proof) = render_scope_free_family_payload_proof(family) else {
                    return Ok(());
                };
                writeln!(branches, "    · exact Mxx.IR.ScopeFreeStep.avoiding (Mxx.IR.evalScope_success_scope_free_family_step linkedProgramData env structural trace {stage_index} stage{stage_index} {} {scope_name} (by rfl) (by rfl) inputs path {index} values fuel finalResult fuelPositive (by decide) {node_name} (by rfl) {node_name}.payload (by rfl) ({family_proof}) success)", scope.id).unwrap();
            }
        }
    }
    writeln!(branches, "    · exact False.elim (sizeMismatch (by decide))").unwrap();
    writeln!(
        source,
        "theorem {loop_name}SuffixAvoids {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (selectedIteration : Nat) (selectedPath : Mxx.IR.OccurrencePath)\n  (selectedUnder : Mxx.IR.OccurrencePath.Under\n    (path.push ⟨{stage_index}, {loop_child}, {loop_index}, selectedIteration⟩) selectedPath) :\n  ∀ (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n    (finalResult : Mxx.IR.ScopeResult backend),\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index}\n      stage{stage_index} {} {scope_name} (by rfl) (by rfl) inputs path {start} values fuel =\n        .ok finalResult →\n    ∀ snapshot ∈ finalResult.scopes, snapshot.occurrence ≠ selectedPath := by\n  intro values fuel finalResult success\n  let selectedFrame : Mxx.IR.OccurrenceFrame :=\n    ⟨{stage_index}, {loop_child}, {loop_index}, selectedIteration⟩\n  have parentDifferent : path ≠ selectedPath :=\n    Mxx.IR.OccurrencePath.ne_parent_of_push_under path selectedPath selectedFrame\n      (by simpa [selectedFrame] using selectedUnder)\n  have suffixStep : ∀ (index : Nat) (values : Array (Mxx.IR.Binding backend))\n      (fuel : Nat) (finalResult : Mxx.IR.ScopeResult backend),\n      {start} ≤ index → index < {scope_name}.nodes.size → fuel ≠ 0 →\n      Mxx.IR.evalScope linkedProgramData env structural trace {stage_index}\n        stage{stage_index} {} {scope_name} (by rfl) (by rfl) inputs path index values fuel =\n          .ok finalResult →\n      Mxx.IR.AvoidingScopeStep linkedProgramData env structural trace {stage_index}\n        stage{stage_index} {} {scope_name} (by rfl) (by rfl) inputs path selectedPath index\n        values fuel finalResult := by\n    intro index values fuel finalResult startBound indexBound fuelPositive success\n    have indexCases : {cases} := by\n      simp [{scope_name}] at indexBound\n      omega\n    rcases indexCases with {patterns}\n{branches}  exact Mxx.IR.evalScope_success_suffix_avoids linkedProgramData env structural trace\n    {stage_index} stage{stage_index} {} {scope_name} (by rfl) (by rfl) inputs path selectedPath\n    parentDifferent {start} suffixStep {start} values fuel finalResult (by omega) success\n",
        scope.id, scope.id, scope.id, scope.id
    )
    .unwrap();
    Ok(())
}

fn render_input_equation(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    scope_name: &str,
) {
    let original_scope_name = scope_name;
    let scope_name = format!("{scope} {original_scope_name}");
    let success = format!(
        "Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel = .ok finalResult"
    );
    writeln!(
        source,
        "theorem {node_name}Equation {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n  (finalResult : Mxx.IR.ScopeResult backend) (fuelPositive : fuel ≠ 0)\n  (success : {success}) :\n  ∃ argumentValues result nextResult,\n    Mxx.IR.resolveArguments {stage_index} {scope} {node_index} values {node_name}.arguments = .ok argumentValues ∧\n    ((∃ binding, inputs[{node_index}]? = some binding ∧ result = Mxx.IR.NodeResult.ofValues #[binding.value]) ∨\n      (inputs[{node_index}]? = none ∧ ∃ value,\n        Mxx.IR.envInput env {stage_index} {scope} {node_index} path {{ scope := {scope}, node := {node_index}, port := 0 }} = .ok value ∧\n        result = Mxx.IR.NodeResult.ofValues #[value])) ∧\n    Mxx.IR.outputTypesMatch {node_name}.outputs.toList result.values.toList = true ∧\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope_name}\n      (by rfl) (by rfl) inputs path ({node_index} + 1)\n      (Mxx.IR.appendNodeBindings {scope} {node_index} values result.values) (fuel - 1) = .ok nextResult ∧\n    finalResult = {{ values := nextResult.values, scopes := result.scopes ++ nextResult.scopes ++ #[{{\n      scope := {scope}, occurrence := path, values := Mxx.IR.appendNodeBindings {scope} {node_index} values result.values }}] }} := by\n  exact Mxx.IR.evalScope_success_input_step linkedProgramData env structural trace {stage_index}\n    stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel\n    finalResult fuelPositive (by decide) {node_name} (by rfl) {node_index} (by rfl) success\n"
    )
    .unwrap();
}

fn render_sampler_equation(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    scope_name: &str,
) {
    let original_scope_name = scope_name;
    let scope_name = format!("{scope} {original_scope_name}");
    let success = format!(
        "Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel = .ok finalResult"
    );
    let mut equation = format!(
        "theorem {node_name}Equation {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n  (finalResult : Mxx.IR.ScopeResult backend) (fuelPositive : fuel ≠ 0)\n  (success : {success}) :\n  ∃ argumentValues sampled nextResult,\n    Mxx.IR.resolveArguments {stage_index} {scope} {node_index} values {node_name}.arguments = .ok argumentValues ∧\n    {node_name}.outputs.mapIdxM (fun port _ =>\n      if hPort : port < {node_name}.outputs.size then\n        let outputType := {node_name}.outputs[port]'hPort\n        have hOutput : {node_name}.outputs[port]? = some outputType := by\n          rw [Array.getElem?_eq_getElem]\n        Mxx.IR.envSample env {stage_index} stage{stage_index} {scope_name} {node_index} {node_name} path\n          {{ scope := {scope}, node := {node_index}, port := port }} outputType\n          (by rfl) (by rfl) (by rfl) hPort hOutput (by simp [Mxx.IR.samplerPayload])\n      else throw (.missingPort {stage_index} {scope} {node_index} port)) = .ok sampled ∧\n    Mxx.IR.outputTypesMatch {node_name}.outputs.toList\n      (Mxx.IR.NodeResult.ofValues sampled).values.toList = true ∧\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope_name}\n      (by rfl) (by rfl) inputs path ({node_index} + 1)\n      (Mxx.IR.appendNodeBindings {scope} {node_index} values sampled) (fuel - 1) = .ok nextResult ∧\n    finalResult = {{ values := nextResult.values, scopes := nextResult.scopes ++ #[{{\n      scope := {scope}, occurrence := path, values := Mxx.IR.appendNodeBindings {scope} {node_index} values sampled }}] }} := by\n  exact Mxx.IR.evalScope_success_sampler_step linkedProgramData env structural trace {stage_index}\n    stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel\n    finalResult fuelPositive (by decide) {node_name} (by rfl) {node_name}.payload (by rfl)\n    (by simp [Mxx.IR.samplerPayload]) success\n"
    );
    equation = equation.replace(
        &format!("envSample env {stage_index} stage{stage_index} {scope_name}"),
        &format!("envSample env {stage_index} stage{stage_index} {original_scope_name}"),
    );
    writeln!(source, "{equation}").unwrap();
}

fn render_artifact_equation(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    scope_name: &str,
    artifact_name: &str,
) {
    let original_scope_name = scope_name;
    let scope_name = format!("{scope} {original_scope_name}");
    let success = format!(
        "Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel = .ok finalResult"
    );
    writeln!(
        source,
        "theorem {node_name}Equation {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n  (finalResult : Mxx.IR.ScopeResult backend) (fuelPositive : fuel ≠ 0)\n  (success : {success}) :\n  ∃ argumentValues link producerTrace producerScope value nextResult,\n    Mxx.IR.resolveArguments {stage_index} {scope} {node_index} values {node_name}.arguments = .ok argumentValues ∧\n    linkedProgramData.artifactLinks[{artifact_name}.index]? = some link ∧\n    (Mxx.IR.Trace.stages trace)[link.producerStage]? = some producerTrace ∧\n    producerTrace.scopes.find? (fun item => item.scope = link.producer.scope) = some producerScope ∧\n    Mxx.IR.lookup producerScope.values link.producer = some value ∧\n    Mxx.IR.outputTypesMatch {node_name}.outputs.toList\n      (Mxx.IR.NodeResult.ofValues #[value]).values.toList = true ∧\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope_name}\n      (by rfl) (by rfl) inputs path ({node_index} + 1)\n      (Mxx.IR.appendNodeBindings {scope} {node_index} values #[value]) (fuel - 1) = .ok nextResult ∧\n    finalResult = {{ values := nextResult.values, scopes := nextResult.scopes ++ #[{{\n      scope := {scope}, occurrence := path, values := Mxx.IR.appendNodeBindings {scope} {node_index} values #[value] }}] }} := by\n  exact Mxx.IR.evalScope_success_artifact_step linkedProgramData env structural trace {stage_index}\n    stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel\n    finalResult fuelPositive (by decide) {node_name} (by rfl) {artifact_name} (by rfl) success\n"
    )
    .unwrap();
}

fn render_primitive_equation(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    scope_name: &str,
    payload_proof: &str,
) {
    let original_scope_name = scope_name;
    let scope_name = format!("{scope} {original_scope_name}");
    let success = format!(
        "Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel = .ok finalResult"
    );
    writeln!(
        source,
        "theorem {node_name}Equation {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n  (finalResult : Mxx.IR.ScopeResult backend) (fuelPositive : fuel ≠ 0)\n  (success : {success}) :\n  ∃ argumentValues result nextResult,\n    Mxx.IR.resolveArguments {stage_index} {scope} {node_index} values {node_name}.arguments = .ok argumentValues ∧\n    Mxx.IR.evalPrimitiveNode backend structural {stage_index} {scope} {node_index} {node_name}.payload\n      argumentValues {node_name}.outputs = .ok result ∧\n    Mxx.IR.outputTypesMatch {node_name}.outputs.toList result.values.toList = true ∧\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope_name}\n      (by rfl) (by rfl) inputs path ({node_index} + 1)\n      (Mxx.IR.appendNodeBindings {scope} {node_index} values result.values) (fuel - 1) = .ok nextResult ∧\n    finalResult = {{ values := nextResult.values, scopes := result.scopes ++ nextResult.scopes ++ #[{{\n      scope := {scope}, occurrence := path, values := Mxx.IR.appendNodeBindings {scope} {node_index} values result.values }}] }} := by\n  exact Mxx.IR.generatedPrimitiveNodeEquation linkedProgramData env structural trace {stage_index}\n    stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel\n    finalResult fuelPositive (by decide) {node_name} (by rfl) {node_name}.payload (by rfl)\n    ({payload_proof}) success\n"
    )
    .unwrap();
}

fn render_subgraph_equation(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    scope_name: &str,
    call_name: &str,
) {
    let original_scope_name = scope_name;
    let scope_name = format!("{scope} {original_scope_name}");
    let success = format!(
        "Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel = .ok finalResult"
    );
    writeln!(
        source,
        "theorem {node_name}Equation {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n  (finalResult : Mxx.IR.ScopeResult backend) (fuelPositive : fuel ≠ 0)\n  (success : {success}) :\n  ∃ argumentValues child, ∃ childStored : Mxx.IR.scopeAt stage{stage_index} {call_name}.child = some child,\n    ∃ childInputs : Array (Mxx.IR.Binding backend), ∃ childResult : Mxx.IR.ScopeResult backend,\n    ∃ childOutputs : Array (Mxx.IR.DynamicValue backend), ∃ nextResult : Mxx.IR.ScopeResult backend,\n    Mxx.IR.resolveArguments {stage_index} {scope} {node_index} values {node_name}.arguments = .ok argumentValues ∧\n    Mxx.IR.checkedChildInputs {stage_index} {scope} {node_index} child argumentValues = .ok childInputs ∧\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {call_name}.child child (by rfl) childStored\n      childInputs (path.push {{ stage := {stage_index}, scope := {scope}, owner := {node_index}, laneOrIteration := 0 }}) 0 #[] (fuel - 1) = .ok childResult ∧\n    child.outputs.mapM (fun output =>\n      (match Mxx.IR.lookup childResult.values output with\n      | some value => Except.ok value\n      | none => Except.error (Mxx.IR.EvalError.missingPort {stage_index} child.id output.node output.port) :\n        Except Mxx.IR.EvalError (Mxx.IR.DynamicValue backend))) = .ok childOutputs ∧\n    Mxx.IR.outputTypesMatch {node_name}.outputs.toList childOutputs.toList = true ∧\n    Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope_name}\n      (by rfl) (by rfl) inputs path ({node_index} + 1)\n      (Mxx.IR.appendNodeBindings {scope} {node_index} values childOutputs) (fuel - 1) = .ok nextResult ∧\n    finalResult = {{ values := nextResult.values, scopes := childResult.scopes ++ nextResult.scopes ++ #[{{\n      scope := {scope}, occurrence := path, values := Mxx.IR.appendNodeBindings {scope} {node_index} values childOutputs }}] }} := by\n  exact Mxx.IR.generatedSubgraphNodeEquation linkedProgramData env structural trace {stage_index}\n    stage{stage_index} {scope} {original_scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel\n    finalResult fuelPositive (by decide) {node_name} (by rfl) {call_name} (by rfl) success\n"
    )
    .unwrap();
}

fn render_grid_equation(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    scope_name: &str,
    grid_name: &str,
) {
    writeln!(
        source,
        "theorem {node_name}Equation {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n  (finalResult : Mxx.IR.ScopeResult backend) (fuelPositive : fuel ≠ 0)\n  (success : Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel = .ok finalResult) :\n  Mxx.IR.ParallelGridEquation linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {scope_name}\n    (by rfl) (by rfl) inputs path {node_index} values fuel {node_name} {grid_name} finalResult := by\n  exact Mxx.IR.generatedParallelGridNodeEquation linkedProgramData env structural trace {stage_index}\n    stage{stage_index} {scope} {scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel\n    finalResult fuelPositive (by decide) {node_name} (by rfl) {grid_name} (by rfl) success\n"
    )
    .unwrap();
}

fn render_loop_equation(
    source: &mut String,
    stage_index: usize,
    scope: usize,
    node_index: usize,
    node_name: &str,
    scope_name: &str,
    loop_name: &str,
) {
    writeln!(
        source,
        "theorem {node_name}Equation {{backend : Mxx.IR.SemanticBackend}}\n  (env : Mxx.IR.EvalEnv backend linkedProgramData)\n  (structural : Mxx.IR.StructuralEnv) (trace : Mxx.IR.Trace backend)\n  (inputs : Array (Mxx.IR.Binding backend)) (path : Mxx.IR.OccurrencePath)\n  (values : Array (Mxx.IR.Binding backend)) (fuel : Nat)\n  (finalResult : Mxx.IR.ScopeResult backend) (fuelPositive : fuel ≠ 0)\n  (success : Mxx.IR.evalScope linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel = .ok finalResult) :\n  Mxx.IR.SequentialLoopEquation linkedProgramData env structural trace {stage_index} stage{stage_index} {scope} {scope_name}\n    (by rfl) (by rfl) inputs path {node_index} values fuel {node_name} {loop_name} finalResult := by\n  exact Mxx.IR.generatedSequentialLoopNodeEquation linkedProgramData env structural trace {stage_index}\n    stage{stage_index} {scope} {scope_name} (by rfl) (by rfl) inputs path {node_index} values fuel\n    finalResult fuelPositive (by decide) {node_name} (by rfl) {loop_name} (by rfl) success\n"
    )
    .unwrap();
}

fn render_node_certificate_scopes(
    ast: &ConcreteLinkedProgram,
) -> Result<Vec<(String, String)>, LeanEmissionError> {
    let mut rendered = Vec::new();
    for (stage_index, stage) in ast.stages.iter().enumerate() {
        for scope in &stage.scopes {
            let scope_name = format!("stage{stage_index}_scope{}", scope.id);
            let mut source = String::new();
            let mut leaf_names = Vec::with_capacity(scope.nodes.len());
            for (node_index, node) in scope.nodes.iter().enumerate() {
                let node_name = format!("{scope_name}_node{node_index}");
                let cert_name = format!("{node_name}Cert");
                let argument_types = node
                    .arguments
                    .iter()
                    .enumerate()
                    .map(|(argument_index, argument)| {
                        scope
                            .nodes
                            .get(argument.node.0 as usize)
                            .and_then(|producer| producer.outputs.get(argument.port.0 as usize))
                            .ok_or_else(|| LeanEmissionError::Encoding {
                                message: format!(
                                    "node {node_index} argument {argument_index} does not resolve in scope {}",
                                    scope.id
                                ),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                leaf_names.push(cert_name.clone());

                let mut output_leaf_names = Vec::with_capacity(node.outputs.len());
                for (output_index, output) in node.outputs.iter().enumerate() {
                    let output_cert = format!("{node_name}Output{output_index}Cert");
                    output_leaf_names.push(output_cert.clone());
                    writeln!(
                        source,
                        "def {output_cert} : Mxx.IR.StoredOutputCert {node_name} {output_index} where\n  output := {}\n  stored := by rfl\n  valid := by\n    simp [Mxx.IR.validWireType, Mxx.IR.MatrixType.Valid, Mxx.IR.structuralExprValid,\n      Mxx.IR.realExprValid]",
                        render_type(output)?
                    )
                    .unwrap();
                }
                render_named_range(
                    &mut source,
                    &format!("{node_name}Outputs"),
                    &format!("Mxx.IR.OutputRangeCert {node_name}"),
                    "Mxx.IR.OutputRangeCert",
                    &output_leaf_names,
                );

                let mut argument_leaf_names = Vec::with_capacity(node.arguments.len());
                for (argument_index, (argument, argument_type)) in
                    node.arguments.iter().zip(&argument_types).enumerate()
                {
                    let argument_cert = format!("{node_name}Argument{argument_index}Cert");
                    argument_leaf_names.push(argument_cert.clone());
                    writeln!(
                        source,
                        "def {argument_cert} : Mxx.IR.StoredArgumentCert {scope_name} {node_index} {node_name} {argument_index} where\n  argument := {}\n  stored := by rfl\n  previous := ⟨rfl, by decide, stage{stage_index}_scope{}_node{}, by rfl, by decide⟩\n  argumentType := {}\n  typeStored := by rfl",
                        render_wire(argument),
                        argument.scope,
                        argument.node.0,
                        render_type(argument_type)?
                    )
                    .unwrap();
                }
                render_named_range(
                    &mut source,
                    &format!("{node_name}Arguments"),
                    &format!("Mxx.IR.ArgumentRangeCert {scope_name} {node_index} {node_name}"),
                    "Mxx.IR.ArgumentRangeCert",
                    &argument_leaf_names,
                );

                let argument_types_name = format!("{node_name}ArgumentTypes");
                let rendered_argument_types = argument_types
                    .iter()
                    .map(|argument_type| {
                        render_type(argument_type).map(|ty| format!("some ({ty})"))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                writeln!(
                    source,
                    "abbrev {argument_types_name} : List (Option Mxx.IR.WireType) := [{rendered_argument_types}]\n"
                )
                .unwrap();
                let mut argument_type_leaf_names = Vec::with_capacity(node.arguments.len());
                for argument_index in 0..node.arguments.len() {
                    let argument_type_cert = format!("{node_name}Argument{argument_index}TypeCert");
                    argument_type_leaf_names.push(argument_type_cert.clone());
                    writeln!(
                        source,
                        "def {argument_type_cert} : Mxx.IR.StoredArgumentTypeCert {scope_name} {node_index} {node_name} {argument_types_name} {argument_index} where\n  argument := {node_name}Argument{argument_index}Cert\n  expected := by\n    simp [{argument_types_name}, {node_name}Argument{argument_index}Cert]"
                    )
                    .unwrap();
                }
                let argument_types_range_name = format!("{node_name}ArgumentTypeRange");
                render_named_range(
                    &mut source,
                    &argument_types_range_name,
                    &format!(
                        "Mxx.IR.ArgumentTypeRangeCert {scope_name} {node_index} {node_name} {argument_types_name}"
                    ),
                    "Mxx.IR.ArgumentTypeRangeCert",
                    &argument_type_leaf_names,
                );
                writeln!(
                    source,
                    "def {node_name}ArgumentsTyped : Mxx.IR.referencedTypes {scope_name} {node_name}.arguments = {argument_types_name} :=\n  {argument_types_range_name}.sound (by rfl)\n"
                )
                .unwrap();

                let output_types_name = format!("{node_name}OutputTypes");
                let rendered_output_types = node
                    .outputs
                    .iter()
                    .map(|output_type| render_type(output_type).map(|ty| format!("some ({ty})")))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                writeln!(
                    source,
                    "abbrev {output_types_name} : List (Option Mxx.IR.WireType) := [{rendered_output_types}]"
                )
                .unwrap();
                writeln!(
                    source,
                    "def {node_name}OutputsTyped : {node_name}.outputs.toList.map some = {output_types_name} := by rfl\n"
                )
                .unwrap();

                let operation_contract = render_operation_contract(
                    &mut source,
                    &node.kind,
                    node,
                    node_index,
                    &node_name,
                    &scope_name,
                    scope,
                    stage,
                    stage_index,
                    &argument_types,
                )?;

                let operation_name = format!("{node_name}Operation");
                writeln!(
                    source,
                    "noncomputable def {operation_name} : Mxx.IR.OperationCert stage{stage_index} {scope_name} {node_index} {node_name}.payload {node_name}.arguments {node_name}.outputs where\n  argumentTypes := {argument_types_name}\n  argumentTypesSize := by rfl\n  argumentsTyped := {argument_types_range_name}\n  contract := {operation_contract}\n"
                )
                .unwrap();

                writeln!(
                    source,
                    "def {node_name}PayloadSlots : Mxx.IR.PayloadSlotsCert {scope_name} {node_name} where\n  valid := by\n    simp [{node_name}, {scope_name}, Mxx.IR.payloadSlotsUsed, Mxx.IR.structuralSlotsUsed,\n      Mxx.IR.indexSlotsUsed, Mxx.IR.realSlotsUsed, Mxx.IR.rangeSlotsUsed,\n      Mxx.IR.mapSlotsUsed, Mxx.IR.slotDeclared]\n"
                )
                .unwrap();
                writeln!(
                    source,
                    "noncomputable def {node_name}LocalCert : Mxx.IR.Node.LocalCert stage{stage_index} {scope_name} {node_index} {node_name} where"
                )
                .unwrap();
                writeln!(source, "  outputsNonempty := by decide").unwrap();
                writeln!(source, "  outputs := {node_name}Outputs").unwrap();
                writeln!(source, "  arguments := {node_name}Arguments").unwrap();
                writeln!(
                    source,
                    "  payload := by\n    simp [{node_name}, Mxx.IR.NodePayload.Valid, Mxx.IR.validPayload,\n      Mxx.IR.MatrixType.Valid, Mxx.IR.structuralExprValid, Mxx.IR.realExprValid,\n      Mxx.IR.optionRangeValid, Mxx.IR.shapeValid, Mxx.IR.indexMapValid,\n      Mxx.IR.indexExprValid]"
                )
                .unwrap();
                writeln!(source, "  payloadSlots := {node_name}PayloadSlots").unwrap();
                writeln!(source, "  operation := {operation_name}\n").unwrap();
                writeln!(
                    source,
                    "def {cert_name} : Mxx.IR.StoredNodeCert stage{stage_index} {scope_name} {node_index} where"
                )
                .unwrap();
                writeln!(source, "  node := {node_name}").unwrap();
                writeln!(source, "  stored := by rfl").unwrap();
                writeln!(source, "  valid := {node_name}LocalCert.sound\n").unwrap();
            }
            render_named_range(
                &mut source,
                &format!("{scope_name}Nodes"),
                &format!("Mxx.IR.NodeRangeCert stage{stage_index} {scope_name}"),
                "Mxx.IR.NodeRangeCert",
                &leaf_names,
            );
            rendered.push((scope_name, source));
        }
    }
    Ok(rendered)
}

#[allow(clippy::too_many_arguments)]
fn render_operation_contract(
    source: &mut String,
    kind: &ConcreteNodePayload,
    node: &ConcreteNode,
    _node_index: usize,
    node_name: &str,
    scope_name: &str,
    scope: &crate::linked::ConcreteScope,
    stage: &crate::linked::ConcreteLinkedStage,
    stage_index: usize,
    argument_types: &[&ConcreteWireType],
) -> Result<String, LeanEmissionError> {
    use ConcreteNodePayload as Kind;
    if matches!(kind, Kind::SubgraphCall(_) | Kind::SequentialLoop(_) | Kind::ParallelGrid(_)) {
        return render_structural_operation_contract(
            source,
            kind,
            node,
            node_name,
            scope_name,
            scope,
            stage,
            stage_index,
            argument_types,
        );
    }
    let (constructor, explicit_arguments) = match kind {
        Kind::Input { artifact: None, .. } => ("input", 1),
        Kind::Input { artifact: Some(_), .. } => ("artifactInput", 1),
        Kind::ConstantInt(_) => ("constantInt", 1),
        Kind::EvaluateInt(_) => ("evaluateInt", 1),
        Kind::ConstantBool(_) => ("constantBool", 1),
        Kind::ConstantMatrix { .. } => ("constantMatrix", 2),
        Kind::UniformResidueSample { .. } => ("uniformResidueSample", 1),
        Kind::UniformIntervalSample { .. } => ("uniformIntervalSample", 2),
        Kind::GaussianSample { .. } => ("gaussianSample", 3),
        Kind::HashSample { .. } => ("hashSample", 5),
        Kind::TrapdoorSample { .. } => ("trapdoorSample", 5),
        Kind::IntBinary(_) => ("intBinary", 1),
        Kind::IntCompare(_) => ("intCompare", 1),
        Kind::BitExtract { .. } => ("bitExtract", 1),
        Kind::RealBinary(_) => ("realBinary", 1),
        Kind::IntToReal => ("intToReal", 0),
        Kind::BoolToInt => ("boolToInt", 0),
        Kind::RealSqrt => ("realSqrt", 0),
        Kind::MatrixBinary(_) => ("matrixBinary", 1),
        Kind::MatrixNegate => ("matrixNegate", 0),
        Kind::MatrixScale { .. } => ("matrixScale", 1),
        Kind::Transpose => ("transpose", 0),
        Kind::Concat { .. } => ("concat", 1),
        Kind::Slice { .. } => ("slice", 2),
        Kind::ExtractCoefficient { .. } => ("extractCoefficient", 2),
        Kind::PreimageSample { .. } => ("preimageSample", 2),
        Kind::FamilyPreimageSample { .. } => ("familyPreimageSample", 2),
        Kind::ApplyPreimage => ("applyPreimage", 0),
        Kind::MaterializePreimageExact => ("materializePreimageExact", 0),
        Kind::PreimageBinary(_) => ("preimageBinary", 1),
        Kind::GadgetDecompose { .. } => ("gadgetDecompose", 3),
        Kind::FamilyPack { .. } => ("familyPack", 1),
        Kind::FamilyGetStatic { .. } => ("familyGetStatic", 1),
        Kind::FamilyGetDynamic { .. } => ("familyGetDynamic", 1),
        Kind::FamilySelectAxis { .. } => ("familySelectAxis", 1),
        Kind::FamilyReindex { .. } => ("familyReindex", 2),
        Kind::FamilyGather { .. } => ("familyGather", 2),
        Kind::Select { .. } => ("select", 1),
        Kind::SubgraphCall(_) | Kind::SequentialLoop(_) | Kind::ParallelGrid(_) => unreachable!(),
        unsupported => {
            return Err(LeanEmissionError::Encoding {
                message: format!(
                    "validated program contains an operation without a declarative certificate: {unsupported:?}"
                ),
            });
        }
    };
    let placeholders = std::iter::repeat_n("_", explicit_arguments).collect::<Vec<_>>().join(" ");
    let validity = match kind {
        Kind::FamilyReindex { map, .. } => {
            let Some(ConcreteWireType::Family { shape: input_shape, element }) =
                argument_types.first().copied()
            else {
                return Err(LeanEmissionError::Encoding {
                    message: "family reindex input is not a family".to_owned(),
                });
            };
            let Some(ConcreteWireType::Family { shape: output_shape, element: output_element }) =
                node.outputs.first()
            else {
                return Err(LeanEmissionError::Encoding {
                    message: "family reindex output is not a family".to_owned(),
                });
            };
            if element != output_element {
                return Err(LeanEmissionError::Encoding {
                    message: "family reindex changes its element type".to_owned(),
                });
            }
            let input_shape = render_nat_list(input_shape);
            let output_shape = render_nat_list(output_shape);
            let element = render_type(element)?;
            let map_checked = render_index_map_checked_proof(map);
            format!(
                "by\n      refine ⟨{input_shape}, {output_shape}, {element}, by rfl, ?_, by rfl, by rfl, by rfl, ?_, by rfl⟩\n      · exact (Mxx.IR.ShapeCert.ofChecked (by rfl)).sound\n      · rw [{scope_name}StructuralSlots]\n        exact (Mxx.IR.IndexMapCert.ofChecked ({map_checked})).sound"
            )
        }
        Kind::Select { .. } => {
            let Some(selector) = argument_types.first().copied() else {
                return Err(LeanEmissionError::Encoding {
                    message: "select has no selector argument".to_owned(),
                });
            };
            let Some(branch) = node.outputs.first() else {
                return Err(LeanEmissionError::Encoding {
                    message: "select has no output".to_owned(),
                });
            };
            let branch_count = argument_types.len().saturating_sub(1);
            format!(
                "by\n      refine ⟨{branch_count}, {}, {}, ?_, by decide, ?_, by rfl, by rfl⟩\n      · exact (Mxx.IR.StructuralNatCert.ofEval (by rfl)).sound\n      · trivial",
                render_type(branch)?,
                render_type(selector)?
            )
        }
        Kind::GadgetDecompose { .. } => {
            let Some(ConcreteWireType::Matrix(target)) = argument_types.first().copied() else {
                return Err(LeanEmissionError::Encoding {
                    message: "gadget decomposition target is not a matrix".to_owned(),
                });
            };
            let Some(ConcreteWireType::Preimage(preimage)) = node.outputs.first() else {
                return Err(LeanEmissionError::Encoding {
                    message: "gadget decomposition output is not a preimage".to_owned(),
                });
            };
            if target.rows == 0 || preimage.rows % target.rows != 0 {
                return Err(LeanEmissionError::Encoding {
                    message: "gadget decomposition has a non-integral digit count".to_owned(),
                });
            }
            let digit_count = preimage.rows / target.rows;
            format!(
                "by\n      refine ⟨_, _, {digit_count}, by rfl, by rfl, ?_, ?_, ?_, by rfl, by rfl⟩\n      · exact (Mxx.IR.StructuralNatCert.ofEval (by rfl)).sound\n      · omega\n      · simp [Mxx.IR.sameRing]"
            )
        }
        Kind::ExtractCoefficient { position, .. } => {
            let crate::linked::ConcreteStructuralIntExpr::Literal(value) = position else {
                return Err(LeanEmissionError::Encoding {
                    message: "extract-coefficient position is not a closed literal".to_owned(),
                });
            };
            format!(
                "by\n      refine ⟨_, {value}, by rfl, by rfl, by rfl, by rfl, by decide, by decide⟩"
            )
        }
        Kind::IntBinary(_) | Kind::IntCompare(_) | Kind::BitExtract { .. } =>
            "by simp [Mxx.IR.scalarIntegerType]".to_owned(),
        Kind::BoolToInt => "by simp [Mxx.IR.scalarBooleanType]".to_owned(),
        Kind::MatrixBinary(_) | Kind::ApplyPreimage | Kind::PreimageBinary(_) =>
            "by simp [Mxx.IR.sameRing, Mxx.IR.matrixAddType, Mxx.IR.matrixProductType]"
                .to_owned(),
        Kind::Transpose => "by simp [Mxx.IR.sameRing]".to_owned(),
        Kind::Concat { .. } =>
            "by simp [Mxx.IR.matrixConcatType, Mxx.IR.sameRing]".to_owned(),
        Kind::Slice { .. } => {
            let Some(ConcreteWireType::Matrix(input)) = argument_types.first().copied() else {
                return Err(LeanEmissionError::Encoding {
                    message: "slice input is not a matrix".to_owned(),
                });
            };
            let Some(ConcreteWireType::Matrix(output)) = node.outputs.first() else {
                return Err(LeanEmissionError::Encoding {
                    message: "slice output is not a matrix".to_owned(),
                });
            };
            format!(
                "by\n      refine ⟨{}, {}, {}, {}, by rfl, by rfl, ?_, ?_, by simp [Mxx.IR.sameRing], by rfl, by rfl, by decide, by decide⟩\n      · rw [{scope_name}StructuralSlots]\n        exact (Mxx.IR.RangeExtentCert.ofChecked (by rfl)).sound\n      · rw [{scope_name}StructuralSlots]\n        exact (Mxx.IR.RangeExtentCert.ofChecked (by rfl)).sound",
                render_matrix_expr(input)?,
                render_matrix_expr(output)?,
                output.rows,
                output.columns
            )
        }
        Kind::PreimageSample { .. } =>
            "by simp [Mxx.IR.preimageEquationType, Mxx.IR.sameRing]".to_owned(),
        Kind::FamilyPreimageSample { .. } => "by simp [Mxx.IR.matrixFamilyElement?, Mxx.IR.trapdoorFamilyElement?, Mxx.IR.preimageEquationType, Mxx.IR.sameRing]".to_owned(),
        Kind::FamilyPack { .. } => "by simp [Mxx.IR.shapeExpressionIs, Mxx.IR.StructuralIntExpr.eval, Mxx.IR.familyElementType]".to_owned(),
        Kind::FamilyGetDynamic { .. } | Kind::FamilyGather { .. } =>
            "by simp [Mxx.IR.allIntegerTypes, Mxx.IR.integerSelectorType, Mxx.IR.shapeExpressionIs, Mxx.IR.StructuralIntExpr.eval]".to_owned(),
        Kind::FamilySelectAxis { .. } =>
            "by simp [Mxx.IR.removeAt?, Mxx.IR.integerSelectorType]".to_owned(),
        _ => "by simp".to_owned(),
    };
    Ok(format!(
        "Mxx.IR.OperationContractCert.direct (Mxx.IR.DirectOperationCert.{constructor} {placeholders} ({validity}))"
    ))
}

#[allow(clippy::too_many_arguments)]
fn render_structural_operation_contract(
    source: &mut String,
    kind: &ConcreteNodePayload,
    node: &ConcreteNode,
    node_name: &str,
    scope_name: &str,
    _scope: &crate::linked::ConcreteScope,
    stage: &crate::linked::ConcreteLinkedStage,
    stage_index: usize,
    argument_types: &[&ConcreteWireType],
) -> Result<String, LeanEmissionError> {
    let child_id = node.child_scope.ok_or_else(|| LeanEmissionError::Encoding {
        message: format!("structural node {node_name} has no child scope"),
    })?;
    let child =
        stage.scopes.iter().find(|candidate| candidate.id == child_id).ok_or_else(|| {
            LeanEmissionError::Encoding {
                message: format!("structural node {node_name} refers to missing child {child_id}"),
            }
        })?;
    let child_name = format!("stage{stage_index}_scope{child_id}");
    let child_inputs = child
        .inputs
        .iter()
        .map(|wire| resolve_scope_wire_type(child, wire))
        .collect::<Result<Vec<_>, _>>()?;
    let child_outputs = child
        .outputs
        .iter()
        .map(|wire| resolve_scope_wire_type(child, wire))
        .collect::<Result<Vec<_>, _>>()?;
    if matches!(kind, ConcreteNodePayload::ParallelGrid(_)) {
        return render_parallel_grid_contract(
            source,
            kind,
            node,
            node_name,
            scope_name,
            child,
            &child_name,
            stage_index,
            argument_types,
        );
    }
    let node_outputs = node.outputs.iter().collect::<Vec<_>>();
    let inputs_name = format!("{node_name}StructuralInputs");
    render_optional_types_certificate(
        source,
        &inputs_name,
        &format!("Mxx.IR.referencedTypes {scope_name} {node_name}.arguments"),
        &format!("Mxx.IR.referencedTypes {child_name} {child_name}.inputs"),
        &format!("{node_name}ArgumentsTyped"),
        &format!("{node_name}ArgumentTypes"),
        &format!("{child_name}InputsTyped"),
        &format!("{child_name}InputTypes"),
        argument_types,
        &child_inputs,
    )?;
    let outputs_name = format!("{node_name}StructuralOutputs");
    render_optional_types_certificate(
        source,
        &outputs_name,
        &format!("Mxx.IR.referencedTypes {child_name} {child_name}.outputs"),
        &format!("{node_name}.outputs.toList.map some"),
        &format!("{child_name}OutputsTyped"),
        &format!("{child_name}OutputTypes"),
        &format!("{node_name}OutputsTyped"),
        &format!("{node_name}OutputTypes"),
        &child_outputs,
        &node_outputs,
    )?;
    match kind {
        ConcreteNodePayload::SubgraphCall(payload) => {
            let call_name = format!("{node_name}Call");
            writeln!(
                source,
                "abbrev {call_name} : Mxx.IR.SubgraphPayload := {}\n",
                render_subgraph_payload(payload, child_id)
            )
            .unwrap();
            Ok(format!(
                "Mxx.IR.OperationContractCert.structural (Mxx.IR.StructuralOperationCert.subgraphCall {call_name} {node_name}.arguments {node_name}.outputs {child_name} {child_name}Stored (by rfl) {inputs_name} {outputs_name})"
            ))
        }
        ConcreteNodePayload::SequentialLoop(payload) => {
            let loop_name = format!("{node_name}Loop");
            writeln!(
                source,
                "abbrev {loop_name} : Mxx.IR.LoopPayload := {}\n",
                render_loop_payload(payload, child_id)
            )
            .unwrap();
            let declaration = child
                .structural_slots
                .iter()
                .find(|declaration| {
                    declaration.slot == payload.index_slot &&
                        matches!(
                            declaration.kind,
                            crate::linked::StructuralSlotKind::SequentialIteration
                        )
                })
                .ok_or_else(|| LeanEmissionError::Encoding {
                    message: format!("loop {node_name} has no matching iteration declaration"),
                })?;
            let carried_name = format!("{node_name}StructuralCarried");
            render_optional_types_certificate(
                source,
                &carried_name,
                &format!(
                    "(Mxx.IR.referencedTypes {scope_name} {node_name}.arguments).take {}",
                    payload.carried_count
                ),
                &format!("{node_name}.outputs.toList.map some"),
                &format!(
                    "(congrArg (fun values => values.take {}) {node_name}ArgumentsTyped)",
                    payload.carried_count
                ),
                &format!("{node_name}ArgumentTypes.take {}", payload.carried_count),
                &format!("{node_name}OutputsTyped"),
                &format!("{node_name}OutputTypes"),
                &argument_types[..payload.carried_count],
                &node_outputs,
            )?;
            let declaration_name = format!("{node_name}IterationDeclaration");
            writeln!(
                source,
                "abbrev {declaration_name} : Mxx.IR.StructuralSlotDecl := {}\n",
                render_structural_slot(declaration)
            )
            .unwrap();
            Ok(format!(
                "Mxx.IR.OperationContractCert.structural (Mxx.IR.StructuralOperationCert.sequentialLoop {loop_name} {node_name}.arguments {node_name}.outputs {child_name} {} {child_name}Stored (Mxx.IR.StructuralNatCert.ofEval (by rfl)) (by decide) (by decide) (by rfl) (by rfl) (by rfl) {inputs_name} {outputs_name} {carried_name} {declaration_name} (by simp [{child_name}, {declaration_name}]) (by rfl) (by rfl) (by rfl))",
                declaration.upper_bound
            ))
        }
        ConcreteNodePayload::ParallelGrid(_) => unreachable!(),
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn render_parallel_grid_contract(
    source: &mut String,
    kind: &ConcreteNodePayload,
    node: &ConcreteNode,
    node_name: &str,
    scope_name: &str,
    child: &crate::linked::ConcreteScope,
    child_name: &str,
    _stage_index: usize,
    argument_types: &[&ConcreteWireType],
) -> Result<String, LeanEmissionError> {
    let ConcreteNodePayload::ParallelGrid(grid) = kind else { unreachable!() };
    let child_id = child.id;
    let grid_name = format!("{node_name}Grid");
    writeln!(
        source,
        "abbrev {grid_name} : Mxx.IR.GridPayload := {}\n",
        render_grid_payload(grid, child_id)
    )
    .unwrap();
    let shape = node
        .outputs
        .first()
        .and_then(|output| match output {
            ConcreteWireType::Family { shape, .. } => Some(shape.clone()),
            _ => None,
        })
        .ok_or_else(|| LeanEmissionError::Encoding {
            message: format!("parallel grid {node_name} has no family output shape"),
        })?;
    let shape_expression =
        format!("[{}]", shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "));
    let child_inputs = child
        .inputs
        .iter()
        .map(|wire| resolve_scope_wire_type(child, wire))
        .collect::<Result<Vec<_>, _>>()?;
    let mut input_leaves = Vec::with_capacity(argument_types.len());
    for (index, ((outer, inner), mode)) in
        argument_types.iter().zip(&child_inputs).zip(&grid.input_modes).enumerate()
    {
        let leaf = format!("{node_name}GridInput{index}");
        input_leaves.push(leaf.clone());
        let certificate = match mode {
            crate::linked::ConcreteGridInputMode::Broadcast => {
                if outer != inner {
                    return Err(LeanEmissionError::Encoding {
                        message: format!("grid {node_name} broadcast input {index} changes type"),
                    });
                }
                format!(
                    "Mxx.IR.GridInputCert.broadcast {index} ({}) ((congrArg (fun values => values[{index}]?) {node_name}ArgumentsTyped).trans (by rfl)) ((congrArg (fun values => values[{index}]?) {child_name}InputsTyped).trans (by rfl)) (by rfl)",
                    render_type(outer)?
                )
            }
            crate::linked::ConcreteGridInputMode::Reindex { map } => {
                let ConcreteWireType::Family { shape: input_shape, element } = outer else {
                    return Err(LeanEmissionError::Encoding {
                        message: format!("grid {node_name} reindex input {index} is not a family"),
                    });
                };
                if element.as_ref() != *inner {
                    return Err(LeanEmissionError::Encoding {
                        message: format!("grid {node_name} reindex input {index} element mismatch"),
                    });
                }
                let input_shape = format!(
                    "[{}]",
                    input_shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
                );
                let map_checked = render_index_map_checked_proof(map);
                let map = render_index_map(map);
                format!(
                    "Mxx.IR.GridInputCert.reindex {index} {input_shape} ({}) ({map}) ((congrArg (fun values => values[{index}]?) {node_name}ArgumentsTyped).trans (by rfl)) ((congrArg (fun values => values[{index}]?) {child_name}InputsTyped).trans (by rfl)) (by rfl) (by rfl) (by rfl) (by rfl) (Mxx.IR.IndexMapCert.ofChecked ({map_checked}))",
                    render_type(element)?
                )
            }
        };
        writeln!(
            source,
            "def {leaf} : Mxx.IR.GridInputCert {scope_name} {child_name} {grid_name} {shape_expression} {node_name}.arguments {index} :=\n  {certificate}\n"
        )
        .unwrap();
    }
    let inputs_range = format!("{node_name}GridInputs");
    render_named_range(
        source,
        &inputs_range,
        &format!(
            "Mxx.IR.DataRangeCert (Mxx.IR.GridInputCert {scope_name} {child_name} {grid_name} {shape_expression} {node_name}.arguments)"
        ),
        "Mxx.IR.DataRangeCert",
        &input_leaves,
    );

    let child_outputs = child
        .outputs
        .iter()
        .map(|wire| resolve_scope_wire_type(child, wire))
        .collect::<Result<Vec<_>, _>>()?;
    let mut output_leaves = Vec::with_capacity(node.outputs.len());
    for (index, (child_output, output)) in child_outputs.iter().zip(&node.outputs).enumerate() {
        let ConcreteWireType::Family { shape: output_shape, element } = output else {
            return Err(LeanEmissionError::Encoding {
                message: format!("parallel grid {node_name} output {index} is not a family"),
            });
        };
        let leaf = format!("{node_name}GridOutput{index}");
        output_leaves.push(leaf.clone());
        if output_shape != &shape || *child_output != element.as_ref() {
            return Err(LeanEmissionError::Encoding {
                message: format!("parallel grid {node_name} output {index} type mismatch"),
            });
        }
        writeln!(
            source,
            "def {leaf} : Mxx.IR.GridOutputCert {child_name} {node_name}.outputs {shape_expression} {index} where\n  childType := {}\n  outputElement := {}\n  childStored := (congrArg (fun values => values[{index}]?) {child_name}OutputsTyped).trans (by rfl)\n  outputStored := by rfl\n  typeEq := by rfl\n",
            render_type(child_output)?,
            render_type(element)?
        )
        .unwrap();
    }
    let outputs_range = format!("{node_name}GridOutputs");
    render_named_range(
        source,
        &outputs_range,
        &format!(
            "Mxx.IR.DataRangeCert (Mxx.IR.GridOutputCert {child_name} {node_name}.outputs {shape_expression})"
        ),
        "Mxx.IR.DataRangeCert",
        &output_leaves,
    );

    let mut axis_leaves = Vec::with_capacity(shape.len());
    for (axis, (&slot, &extent)) in grid.index_slots.iter().zip(&shape).enumerate() {
        let declaration = child
            .structural_slots
            .iter()
            .find(|declaration| {
                declaration.slot == slot &&
                    matches!(
                        declaration.kind,
                        crate::linked::StructuralSlotKind::GridAxis { axis: found } if found == axis
                    )
            })
            .ok_or_else(|| LeanEmissionError::Encoding {
                message: format!("parallel grid {node_name} has no declaration for axis {axis}"),
            })?;
        let leaf = format!("{node_name}GridAxis{axis}");
        axis_leaves.push(leaf.clone());
        writeln!(
            source,
            "def {leaf} : Mxx.IR.GridAxisCert {child_name} {grid_name} {shape_expression} {axis} where\n  slot := {slot}\n  extent := {extent}\n  declaration := {}\n  slotStored := by rfl\n  extentStored := by rfl\n  declarationMem := by simp [{child_name}]\n  declarationSlot := by rfl\n  declarationKind := by rfl\n  declarationBound := by rfl\n",
            render_structural_slot(declaration)
        )
        .unwrap();
    }
    let axes_range = format!("{node_name}GridAxes");
    render_named_range(
        source,
        &axes_range,
        &format!(
            "Mxx.IR.DataRangeCert (Mxx.IR.GridAxisCert {child_name} {grid_name} {shape_expression})"
        ),
        "Mxx.IR.DataRangeCert",
        &axis_leaves,
    );
    Ok(format!(
        "Mxx.IR.OperationContractCert.structural (Mxx.IR.StructuralOperationCert.parallelGrid {grid_name} {node_name}.arguments {node_name}.outputs {child_name} {shape_expression} {child_name}Stored (Mxx.IR.ShapeCert.ofChecked (by rfl)) (by rfl) (by rfl) (by rfl) (by rfl) {inputs_range} {outputs_range} {axes_range})"
    ))
}

fn render_payload(
    node: &ConcreteNode,
    node_index: usize,
    stage: &crate::linked::ConcreteLinkedStage,
    stage_index: usize,
    scope: usize,
    program: &ConcreteLinkedProgram,
) -> Result<String, LeanEmissionError> {
    use ConcreteNodePayload as NodeKind;
    let input_index = node_index;
    Ok(match &node.kind {
        NodeKind::Input { artifact: Some(artifact), .. } => {
            format!(
                ".artifactInput {{ index := {}, name := {}, confidentiality := {} }}",
                artifact_index(program, stage_index, scope, node_index)?,
                lean_string(&artifact.name),
                render_confidentiality(artifact.confidentiality)
            )
        }
        NodeKind::Input { .. } => format!(".input {}", input_index),
        NodeKind::ConstantInt(value) => format!(".constantInt {}", value),
        NodeKind::ConstantBool(value) => format!(".constantBool {}", value),
        NodeKind::ConstantReal(value) => {
            format!(".constantReal {}", render_real(value))
        }
        NodeKind::EvaluateInt(value) => format!(".evaluateInt {}", render_int(value)),
        NodeKind::ConstantMatrix { matrix_type, value } => format!(
            ".constantMatrix {} ({})",
            render_matrix_expr(matrix_type)?,
            render_constant_matrix(value)
        ),
        NodeKind::GadgetTrapdoor { matrix_type, base } => {
            format!(".gadgetTrapdoor {} {}", render_matrix_expr(matrix_type)?, render_int(base))
        }
        NodeKind::TrapdoorPublic => ".trapdoorPublic".to_owned(),
        NodeKind::IntBinary(crate::node::IntBinaryOp::Add) => ".intBinary .add".to_owned(),
        NodeKind::IntBinary(crate::node::IntBinaryOp::Subtract) => {
            ".intBinary .subtract".to_owned()
        }
        NodeKind::IntBinary(crate::node::IntBinaryOp::Multiply) => {
            ".intBinary .multiply".to_owned()
        }
        NodeKind::IntBinary(crate::node::IntBinaryOp::Divide) => ".intBinary .divide".to_owned(),
        NodeKind::IntBinary(crate::node::IntBinaryOp::Remainder) => {
            ".intBinary .remainder".to_owned()
        }
        NodeKind::IntCompare(op) => format!(
            ".intCompare {}",
            match op {
                crate::node::IntCompareOp::Equal => ".equal",
                crate::node::IntCompareOp::Less => ".less",
                crate::node::IntCompareOp::LessEqual => ".lessEqual",
            }
        ),
        NodeKind::BitExtract { bit } => format!(".bitExtract {}", render_int(bit)),
        NodeKind::IntToReal => ".intToReal".to_owned(),
        NodeKind::BoolToInt => ".boolToInt".to_owned(),
        NodeKind::RealBinary(op) => format!(".realBinary {}", render_real_binary(*op)),
        NodeKind::RealSqrt => ".realSqrt".to_owned(),
        NodeKind::MatrixBinary(op) => format!(
            ".matrixBinary {}",
            match op {
                crate::node::MatrixBinaryOp::Add => ".add",
                crate::node::MatrixBinaryOp::Subtract => ".subtract",
                crate::node::MatrixBinaryOp::Multiply => ".multiply",
            }
        ),
        NodeKind::MatrixNegate => ".matrixNegate".to_owned(),
        NodeKind::MatrixScale { scalar } => format!(".matrixScale {}", render_int(scalar)),
        NodeKind::MatrixMulAccumulate { coefficients, has_bias } => format!(
            ".matrixMulAccumulate #[{}] {}",
            coefficients.iter().map(render_int).collect::<Vec<_>>().join(", "),
            has_bias
        ),
        NodeKind::Transpose => ".transpose".to_owned(),
        NodeKind::Slice { rows, columns } => {
            format!(".slice ({}) ({})", render_range(rows), render_range(columns))
        }
        NodeKind::Tensor => ".tensor".to_owned(),
        NodeKind::Concat { axis } => format!(
            ".concat {}",
            match axis {
                crate::node::ConcatAxis::Rows => ".rows",
                crate::node::ConcatAxis::Columns => ".columns",
                crate::node::ConcatAxis::Diagonal => ".diagonal",
            }
        ),
        NodeKind::UniformResidueSample { matrix_type } => {
            format!(".uniformResidueSample {}", render_matrix_expr(matrix_type)?)
        }
        NodeKind::UniformIntervalSample { matrix_type, range } => format!(
            ".uniformIntervalSample {} {{ start := {}, stop := {} }}",
            render_matrix_expr(matrix_type)?,
            render_int(&range.minimum),
            render_int(&range.maximum)
        ),
        NodeKind::GaussianSample { matrix_type, sigma, max_coefficient_bound } => format!(
            ".gaussianSample {} {} {}",
            render_matrix_expr(matrix_type)?,
            render_real(sigma),
            render_int(max_coefficient_bound)
        ),
        NodeKind::HashSample {
            matrix_type,
            tag_prefix,
            tag_expressions,
            tag_decimal_expressions,
            tag_u64_le_expressions,
        } => {
            format!(
                ".hashSample {} {} #[{}] #[{}] #[{}]",
                render_matrix_expr(matrix_type)?,
                bytes_list(tag_prefix),
                tag_expressions.iter().map(render_int).collect::<Vec<_>>().join(", "),
                tag_decimal_expressions.iter().map(render_int).collect::<Vec<_>>().join(", "),
                tag_u64_le_expressions.iter().map(render_int).collect::<Vec<_>>().join(", ")
            )
        }
        NodeKind::TrapdoorSample {
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => format!(
            ".trapdoorSample {} {} {} {} {}",
            render_matrix_expr(matrix_type)?,
            render_real(sigma),
            render_int(gadget_base),
            render_int(digit_count),
            render_int(preimage_max_coefficient_bound)
        ),
        NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => format!(
            ".preimageSample {} {}",
            render_matrix_expr(matrix_type)?,
            render_int(max_coefficient_bound)
        ),
        NodeKind::ApplyPreimage => ".applyPreimage".to_owned(),
        NodeKind::MaterializePreimageExact => ".materializePreimageExact".to_owned(),
        NodeKind::PreimageBinary(op) => format!(".preimageBinary {}", render_preimage_binary(*op)),
        NodeKind::PreimageConcatColumns => ".preimageConcatColumns".to_owned(),
        NodeKind::FamilyPreimageSample { matrix_type, max_coefficient_bound } => format!(
            ".familyPreimageSample {} {}",
            render_matrix_expr(matrix_type)?,
            render_int(max_coefficient_bound)
        ),
        NodeKind::GadgetDecompose { base, small, digit_count } => {
            format!(".gadgetDecompose {} {} {}", render_int(base), small, render_int(digit_count))
        }
        NodeKind::DecompositionEntry { row, column } => {
            format!(".decompositionEntry {} {}", render_int(row), render_int(column))
        }
        NodeKind::ExtractCoefficient { position, canonical_input_exclusive_upper } => format!(
            ".extractCoefficient {} {}",
            render_int(position),
            render_optional_big(canonical_input_exclusive_upper)
        ),
        NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
            format!(".liftIntegerToConstantPolynomial {}", render_matrix_expr(matrix_type)?)
        }
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => format!(
            ".thresholdDecode {} {} {}",
            render_int(plaintext_modulus),
            render_int(length),
            output_bool
        ),
        NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => format!(
            ".crtRecompose #[{}] #[{}]",
            render_ints(plaintext_moduli),
            render_ints(reconstruction_coefficients)
        ),
        NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => format!(
            ".packPolynomialCoefficients {} {}",
            render_matrix_expr(matrix_type)?,
            render_int(coefficient_bits)
        ),
        NodeKind::FamilyPack { shape } => {
            format!(".familyPack #[{}]", render_ints(shape))
        }
        NodeKind::FamilyGetStatic { indices } => format!(
            ".familyGetStatic #[{}]",
            indices.iter().map(render_index).collect::<Vec<_>>().join(", ")
        ),
        NodeKind::FamilyGetDynamic { rank } => format!(".familyGetDynamic {}", rank),
        NodeKind::FamilySelectAxis { axis } => format!(".familySelectAxis {}", axis),
        NodeKind::FamilyReindex { output_shape, map } => {
            format!(".familyReindex #[{}] {}", render_ints(output_shape), render_index_map(map))
        }
        NodeKind::FamilyGather { output_shape, input_rank } => {
            format!(".familyGather #[{}] {}", render_ints(output_shape), input_rank)
        }
        NodeKind::Select { count } => format!(".select {}", render_int(count)),
        NodeKind::SubgraphCall(payload) => format!(
            ".subgraphCall {}",
            render_subgraph_payload(payload, child_scope(node, stage, scope)?)
        ),
        NodeKind::SequentialLoop(payload) => format!(
            ".sequentialLoop {}",
            render_loop_payload(payload, child_scope(node, stage, scope)?)
        ),
        NodeKind::ParallelGrid(payload) => format!(
            ".parallelGrid {}",
            render_grid_payload(payload, child_scope(node, stage, scope)?)
        ),
    })
}

fn artifact_index(
    program: &ConcreteLinkedProgram,
    stage: usize,
    scope: usize,
    node: usize,
) -> Result<usize, LeanEmissionError> {
    program
        .artifact_links
        .iter()
        .position(|link| {
            link.consumer_stage == stage &&
                link.consumer.scope == scope &&
                link.consumer.node == crate::types::NodeId(node as u64)
        })
        .ok_or_else(|| LeanEmissionError::Encoding {
            message: format!(
                "artifact input at stage {stage}, scope {scope}, node {node} has no resolved link"
            ),
        })
}

fn child_scope(
    node: &ConcreteNode,
    stage: &crate::linked::ConcreteLinkedStage,
    scope: usize,
) -> Result<usize, LeanEmissionError> {
    node.child_scope.ok_or_else(|| LeanEmissionError::Encoding {
        message: format!(
            "stage {:?}, scope {}: structural node has no child scope",
            stage.key, scope
        ),
    })
}
fn render_binding_exprs(bindings: &[(String, crate::linked::ConcreteStructuralIntExpr)]) -> String {
    bindings
        .iter()
        .map(|(name, value)| format!("({}, {})", lean_string(name), render_int(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_subgraph_payload(
    payload: &crate::linked::ConcreteSubgraphPayload,
    child: usize,
) -> String {
    format!(
        "{{ child := {child}, definition := {}, bindings := #[{}], canonicalInputExclusiveUppers := #[{}] }}",
        lean_string(&payload.definition),
        render_binding_exprs(&payload.bindings),
        payload
            .canonical_input_exclusive_uppers
            .iter()
            .map(render_optional_big)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_loop_payload(payload: &crate::linked::ConcreteSequentialLoop, child: usize) -> String {
    format!(
        "{{ child := {child}, count := {}, indexSlot := {}, bindings := #[{}], carriedCount := {} }}",
        render_int(&payload.count),
        payload.index_slot,
        render_binding_exprs(&payload.bindings),
        payload.carried_count
    )
}

fn render_grid_payload(payload: &crate::linked::ConcreteParallelGrid, child: usize) -> String {
    format!(
        "{{ child := {child}, shape := #[{}], indexSlots := #[{}], bindings := #[{}], inputModes := #[{}] }}",
        render_ints(&payload.shape),
        payload.index_slots.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        render_binding_exprs(&payload.bindings),
        payload.input_modes.iter().map(render_input_mode).collect::<Vec<_>>().join(", ")
    )
}
fn render_input_mode(mode: &crate::linked::ConcreteGridInputMode) -> String {
    match mode {
        crate::linked::ConcreteGridInputMode::Broadcast => {
            "({ reindex := false, map := none })".into()
        }
        crate::linked::ConcreteGridInputMode::Reindex { map } => {
            format!("({{ reindex := true, map := some {} }})", render_index_map(map))
        }
    }
}

fn render_confidentiality(
    confidentiality: crate::artifact::ArtifactConfidentiality,
) -> &'static str {
    match confidentiality {
        crate::artifact::ArtifactConfidentiality::Public => ".Public",
        crate::artifact::ArtifactConfidentiality::Private => ".Private",
    }
}

fn render_structural_slot(slot: &crate::linked::StructuralSlotDecl) -> String {
    let kind = match slot.kind {
        crate::linked::StructuralSlotKind::SequentialIteration => ".sequentialIteration".to_owned(),
        crate::linked::StructuralSlotKind::GridAxis { axis } => format!("(.gridAxis {})", axis),
    };
    format!("{{ slot := {}, kind := {}, upperBound := {} }}", slot.slot, kind, slot.upper_bound)
}

fn render_type(ty: &ConcreteWireType) -> Result<String, LeanEmissionError> {
    match ty {
        ConcreteWireType::ConstantInt => Ok(".constantInt".to_owned()),
        ConcreteWireType::ConstantReal => Ok(".constantReal".to_owned()),
        ConcreteWireType::ConstantBool => Ok(".constantBool".to_owned()),
        ConcreteWireType::Int => Ok(".int".to_owned()),
        ConcreteWireType::Real => Ok(".real".to_owned()),
        ConcreteWireType::Bool => Ok(".bool".to_owned()),
        ConcreteWireType::Bytes { length } => Ok(format!(".bytes {}", length)),
        ConcreteWireType::TypedBlob { type_name, schema_hash } => {
            Ok(format!(".typedBlob {} {}", lean_string(type_name), bytes(schema_hash)))
        }
        ConcreteWireType::Matrix(m) => {
            matrix("matrix", &m.modulus, m.ring_dimension, m.rows, m.columns)
        }
        ConcreteWireType::Preimage(m) => {
            matrix("preimage", &m.modulus, m.ring_dimension, m.rows, m.columns)
        }
        ConcreteWireType::Trapdoor {
            matrix: matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => Ok(format!(
            ".trapdoor {{ matrix := {}, sigma := {}, gadgetBase := {}, digitCount := {}, preimageMaxCoefficientBound := {} }}",
            render_matrix_expr(matrix_type)?,
            render_existing_real(sigma)?,
            format!("(IR.StructuralIntExpr.literal ({} : Int))", gadget_base),
            format!("(IR.StructuralIntExpr.literal ({} : Int))", digit_count),
            format!("(IR.StructuralIntExpr.literal ({} : Int))", preimage_max_coefficient_bound)
        )),
        ConcreteWireType::Family { shape, element } => Ok(format!(
            ".family [{}] ({})",
            shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
            render_type(element)?
        )),
    }
}

fn render_int(expr: &crate::linked::ConcreteStructuralIntExpr) -> String {
    use crate::linked::ConcreteStructuralIntExpr as E;
    match expr {
        E::Literal(value) => format!("(IR.StructuralIntExpr.literal ({} : Int))", value),
        E::StructuralSlot(slot) => format!("(IR.StructuralIntExpr.structuralSlot {})", slot),
        E::Add(lhs, rhs) => {
            format!("(IR.StructuralIntExpr.add {} {})", render_int(lhs), render_int(rhs))
        }
        E::Sub(lhs, rhs) => {
            format!("(IR.StructuralIntExpr.subtract {} {})", render_int(lhs), render_int(rhs))
        }
        E::Mul(lhs, rhs) => {
            format!("(IR.StructuralIntExpr.multiply {} {})", render_int(lhs), render_int(rhs))
        }
        E::ExactDivide(lhs, rhs) => {
            format!("(IR.StructuralIntExpr.exactDivide {} {})", render_int(lhs), render_int(rhs))
        }
        E::RoundDivide(lhs, rhs) => {
            format!("(IR.StructuralIntExpr.roundDivide {} {})", render_int(lhs), render_int(rhs))
        }
        E::Log2Ceil(value) => format!("(IR.StructuralIntExpr.log2Ceil {})", render_int(value)),
    }
}

fn render_existing_real(expr: &crate::expr::RealExpr) -> Result<String, LeanEmissionError> {
    use crate::expr::RealExpr as E;
    Ok(match expr {
        E::Rational(value) => format!(
            "(.literal {{ numerator := {}, denominator := {} }})",
            value.numerator(),
            value.denominator()
        ),
        E::Var(name) => {
            return Err(LeanEmissionError::Encoding {
                message: format!("unclosed real parameter {name}"),
            })
        }
        E::FromInt(_) => {
            return Err(LeanEmissionError::Encoding {
                message: "trapdoor sigma contains an unresolved integer expression".to_owned(),
            })
        }
        E::Add(lhs, rhs) => {
            format!("(.add {} {})", render_existing_real(lhs)?, render_existing_real(rhs)?)
        }
        E::Sub(lhs, rhs) => {
            format!("(.subtract {} {})", render_existing_real(lhs)?, render_existing_real(rhs)?)
        }
        E::Mul(lhs, rhs) => {
            format!("(.multiply {} {})", render_existing_real(lhs)?, render_existing_real(rhs)?)
        }
        E::Div(lhs, rhs) => {
            format!("(.divide {} {})", render_existing_real(lhs)?, render_existing_real(rhs)?)
        }
        E::Sqrt(value) => format!("(.sqrt {})", render_existing_real(value)?),
    })
}

fn render_index(expr: &crate::linked::ConcreteIndexMapExpr) -> String {
    use crate::linked::ConcreteIndexMapExpr as E;
    match expr {
        E::Literal(value) => format!("(IR.IndexMapExpr.literal ({} : Int))", value),
        E::Axis(axis) => format!("(IR.IndexMapExpr.axis {})", axis),
        E::StructuralSlot(slot) => format!("(IR.IndexMapExpr.structuralSlot {})", slot),
        E::Add(lhs, rhs) => {
            format!("(IR.IndexMapExpr.add {} {})", render_index(lhs), render_index(rhs))
        }
        E::Sub(lhs, rhs) => {
            format!("(IR.IndexMapExpr.sub {} {})", render_index(lhs), render_index(rhs))
        }
        E::Mul(lhs, rhs) => {
            format!("(IR.IndexMapExpr.mul {} {})", render_index(lhs), render_index(rhs))
        }
        E::EuclideanDivide(lhs, rhs) => {
            format!("(IR.IndexMapExpr.divide {} {})", render_index(lhs), render_index(rhs))
        }
        E::EuclideanRemainder(lhs, rhs) => {
            format!("(IR.IndexMapExpr.remainder {} {})", render_index(lhs), render_index(rhs))
        }
        E::Equal(lhs, rhs) => {
            format!("(IR.IndexMapExpr.equal {} {})", render_index(lhs), render_index(rhs))
        }
        E::Less(lhs, rhs) => {
            format!("(IR.IndexMapExpr.less {} {})", render_index(lhs), render_index(rhs))
        }
        E::LessEqual(lhs, rhs) => {
            format!("(IR.IndexMapExpr.lessEqual {} {})", render_index(lhs), render_index(rhs))
        }
        E::Log2Ceil(value) => format!("(IR.IndexMapExpr.log2Ceil {})", render_index(value)),
        E::Select { selector, branches } => format!(
            "(IR.IndexMapExpr.select {} #[{}])",
            render_index(selector),
            branches.iter().map(render_index).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn render_real(expr: &crate::linked::ConcreteRealExpr) -> String {
    use crate::linked::ConcreteRealExpr as E;
    match expr {
        E::Rational(value) => format!(
            "(.literal {{ numerator := {}, denominator := {} }})",
            value.numerator(),
            value.denominator()
        ),
        E::FromInt(value) => format!("(.fromInt {})", render_int(value)),
        E::Add(lhs, rhs) => format!("(.add {} {})", render_real(lhs), render_real(rhs)),
        E::Sub(lhs, rhs) => format!("(.subtract {} {})", render_real(lhs), render_real(rhs)),
        E::Mul(lhs, rhs) => format!("(.multiply {} {})", render_real(lhs), render_real(rhs)),
        E::Div(lhs, rhs) => format!("(.divide {} {})", render_real(lhs), render_real(rhs)),
        E::Sqrt(value) => format!("(.sqrt {})", render_real(value)),
    }
}

fn render_matrix_expr(
    expr: &crate::types::ConcreteMatrixType,
) -> Result<String, LeanEmissionError> {
    if expr.modulus <= BigInt::from(1) ||
        expr.ring_dimension == 0 ||
        expr.rows == 0 ||
        expr.columns == 0
    {
        return Err(LeanEmissionError::Encoding { message: "invalid matrix shape".to_owned() });
    }
    Ok(format!(
        "{{ modulus := {}, ringDimension := {}, rows := {}, columns := {} }}",
        expr.modulus, expr.ring_dimension, expr.rows, expr.columns
    ))
}

fn render_constant_matrix(value: &crate::linked::ConcreteMatrixLiteral) -> String {
    use crate::linked::ConcreteMatrixLiteral;
    match value {
        ConcreteMatrixLiteral::Zero => ".zero".into(),
        ConcreteMatrixLiteral::Identity => ".identity".into(),
        ConcreteMatrixLiteral::UnitRow { index } => format!(".unitRow {}", render_int(index)),
        ConcreteMatrixLiteral::UnitColumn { index } => format!(".unitColumn {}", render_int(index)),
        ConcreteMatrixLiteral::Gadget { base, small } => {
            format!(".gadget {} {}", render_int(base), small)
        }
        ConcreteMatrixLiteral::PowerOfBase { base, exponent } => {
            format!(".powerOfBase {} {}", render_int(base), render_int(exponent))
        }
        ConcreteMatrixLiteral::Rotation { exponent } => {
            format!(".rotation {}", render_int(exponent))
        }
        ConcreteMatrixLiteral::Polynomial { coefficients } => format!(
            ".polynomial #[{}]",
            coefficients.iter().map(render_int).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn render_range(range: &Option<crate::linked::ConcreteIndexRange>) -> String {
    range.as_ref().map_or_else(
        || "none".into(),
        |r| format!("some {{ start := {}, stop := {} }}", render_int(&r.start), render_int(&r.end)),
    )
}
fn render_ints(xs: &[crate::linked::ConcreteStructuralIntExpr]) -> String {
    xs.iter().map(render_int).collect::<Vec<_>>().join(", ")
}
fn render_real_binary(op: crate::node::RealBinaryOp) -> &'static str {
    match op {
        crate::node::RealBinaryOp::Add => ".add",
        crate::node::RealBinaryOp::Subtract => ".subtract",
        crate::node::RealBinaryOp::Multiply => ".multiply",
        crate::node::RealBinaryOp::Divide => ".divide",
    }
}
fn render_preimage_binary(op: crate::node::PreimageBinaryOp) -> &'static str {
    match op {
        crate::node::PreimageBinaryOp::Add => ".add",
        crate::node::PreimageBinaryOp::RightMultiplyExact => ".rightMultiplyExact",
        crate::node::PreimageBinaryOp::ComposeExactDecomposition => ".composeExactDecomposition",
    }
}
fn render_optional_big(v: &Option<num_bigint::BigUint>) -> String {
    v.as_ref().map_or_else(|| "none".into(), |x| format!("some {}", x))
}
fn bytes_list(values: &[u8]) -> String {
    format!("[{}]", values.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
}
fn render_index_map(map: &crate::linked::ConcreteIndexMap) -> String {
    format!(
        "{{ sourceRank := {}, outputRank := {}, inputIndices := #[{}] }}",
        map.source_rank,
        map.output_rank,
        map.input_indices.iter().map(render_index).collect::<Vec<_>>().join(", ")
    )
}

fn matrix(
    tag: &str,
    modulus: &BigInt,
    ring: usize,
    rows: usize,
    columns: usize,
) -> Result<String, LeanEmissionError> {
    if *modulus <= BigInt::from(1) || ring == 0 || rows == 0 || columns == 0 {
        return Err(LeanEmissionError::Encoding { message: "invalid matrix shape".to_owned() });
    }
    Ok(format!(
        ".{} {{ modulus := {}, ringDimension := {}, rows := {}, columns := {} }}",
        tag, modulus, ring, rows, columns
    ))
}
fn render_wire(wire: &ConcreteWireRef) -> String {
    format!("{{ scope := {}, node := {}, port := {} }}", wire.scope, wire.node.0, wire.port.0)
}

/// Render one validated structural child-input hop as a Lean value.
///
/// Scope names are converted through the concrete stage table, preserving the
/// same scope numbering used by the generated `Stage`.  The hop itself is not
/// re-derived or normalized by the renderer.
pub fn render_child_input_hop(
    stage: &ConcreteLinkedStage,
    hop: &ChildInputHop,
) -> Result<String, LeanEmissionError> {
    let parent_scope =
        stage.scope_ids.iter().position(|scope| scope == &hop.parent_scope).ok_or_else(|| {
            LeanEmissionError::Encoding {
                message: format!(
                    "child-input hop parent scope {:?} is not in the stage",
                    hop.parent_scope
                ),
            }
        })?;
    Ok(format!(
        "{{ parentScope := {}, owner := {}, inputIndex := {} }}",
        parent_scope, hop.owner.0, hop.input_index
    ))
}

/// Render an explicit child-input path as a Lean array value.
pub fn render_child_input_path(
    stage: &ConcreteLinkedStage,
    path: &[ChildInputHop],
) -> Result<String, LeanEmissionError> {
    path.iter()
        .map(|hop| render_child_input_hop(stage, hop))
        .collect::<Result<Vec<_>, _>>()
        .map(|hops| format!("#[{}]", hops.join(", ")))
}

/// Render one exact parallel-grid body-output exit.
pub fn render_parallel_output_hop(
    stage: &ConcreteLinkedStage,
    hop: &ParallelOutputHop,
) -> Result<String, LeanEmissionError> {
    let parent_scope =
        stage.scope_ids.iter().position(|scope| scope == &hop.parent_scope).ok_or_else(|| {
            LeanEmissionError::Encoding {
                message: format!(
                    "parallel-output hop parent scope {:?} is not in the stage",
                    hop.parent_scope
                ),
            }
        })?;
    Ok(format!(
        "{{ parentScope := {}, owner := {}, outputIndex := {} }}",
        parent_scope, hop.owner.0, hop.output_index
    ))
}

/// Render a structural route in its canonical exit-then-enter order.
pub fn render_structural_value_route(
    stage: &ConcreteLinkedStage,
    route: &StructuralValueRoute,
) -> Result<String, LeanEmissionError> {
    let exits = route
        .exits
        .iter()
        .map(|hop| render_parallel_output_hop(stage, hop))
        .collect::<Result<Vec<_>, _>>()?;
    let enters = route
        .enters
        .iter()
        .map(|hop| render_child_input_hop(stage, hop))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(format!("{{ exits := #[{}], enters := #[{}] }}", exits.join(", "), enters.join(", ")))
}
fn bytes(values: &[u8; 32]) -> String {
    format!("[{}]", values.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "))
}
fn lean_string(value: &str) -> String {
    serde_json::to_string(value).expect("string serialization")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Graph, GraphOutput, IndexExpr, IntExpr, NodeHandle, ParamEnv, SubgraphHandle,
        artifact::{ProductionId, export_validated_manifest},
        linked::{LinkedProgramStage, ValidatedLinkedProgram},
        node::{
            ConstantMatrix, GridInputMode, MatrixBinaryOp, NodeKind, ParallelGrid, SequentialLoop,
        },
        types::{MatrixType, WireType},
        validate, validate_with_manifests, with_new_construction_scope,
    };

    #[test]
    fn certificate_ranges_are_chunked_and_balanced_for_thousands_of_nodes() {
        let names = (0..4096).map(|index| format!("node{index}Cert")).collect::<Vec<_>>();
        let chunks = certificate_chunks("Mxx.IR.NodeRangeCert", &names);
        assert_eq!(chunks.len(), 256);
        assert!(chunks.iter().all(|chunk| chunk.end - chunk.start <= CERTIFICATE_CHUNK_SIZE));
        assert!(chunks.iter().all(|chunk| chunk.depth <= 5));

        let complete = balanced_range_expression("Mxx.IR.NodeRangeCert", &chunks, 0);
        assert_eq!((complete.start, complete.end), (0, 4096));
        assert!(complete.depth <= 13);
        assert!(complete.source.len() < 400_000);

        let mut source = String::new();
        render_named_range(
            &mut source,
            "largeScopeNodes",
            "Mxx.IR.NodeRangeCert stage scope",
            "Mxx.IR.NodeRangeCert",
            &names,
        );
        assert_eq!(source.matches("def largeScopeNodesChunk").count(), 256);
        assert!(source.contains("def largeScopeNodes : Mxx.IR.NodeRangeCert stage scope 0 4096"));
        assert!(source.len() < 500_000);
    }

    #[test]
    fn generated_proofs_do_not_use_spread_or_sequence_semicolon_tactics() {
        let forbidden =
            [["<", ";>"].concat(), [";", " rfl"].concat(), ["constructor", " ;"].concat()];
        for source in
            [include_str!("render.rs"), include_str!("../../../we/src/diamond/correctness/emit.rs")]
        {
            for pattern in &forbidden {
                assert!(!source.contains(pattern), "generated proof uses forbidden `{pattern}`");
            }
        }
    }

    #[test]
    fn generated_proposition_disjunctions_and_rcases_patterns_use_distinct_tokens() {
        assert_eq!(alternatives(["left".to_owned(), "right".to_owned()]), "left ∨ right");

        let scope = crate::linked::ConcreteScope {
            id: 0,
            structural_slots: vec![
                crate::linked::StructuralSlotDecl {
                    slot: 0,
                    kind: crate::linked::StructuralSlotKind::GridAxis { axis: 0 },
                    upper_bound: 1.into(),
                },
                crate::linked::StructuralSlotDecl {
                    slot: 1,
                    kind: crate::linked::StructuralSlotKind::SequentialIteration,
                    upper_bound: 2.into(),
                },
            ],
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
        };
        let proof = render_scope_slots_proof("scope", &scope);
        assert!(proof.contains("have cases : (first = 0 ∧ second = 1) ∨ (first = 1 ∧ second = 0)"));
        assert!(proof.contains("rcases cases with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩"));
    }

    #[test]
    fn structural_children_keep_first_node_occurrence_order() {
        assert_eq!(first_occurrence_unique([1, 2, 19, 2, 7, 19]), [1, 2, 19, 7]);
    }

    #[test]
    fn artifact_uniqueness_branches_separate_matching_equality_from_contradictions() {
        let source = include_str!("render.rs");
        assert!(source.contains("if found_index == link_index"));
        assert!(source.contains("proof.push_str(\"          rfl\\n\")"));
        assert!(
            source.contains(
                "proof.push_str(\"          simp at consumerStage consumer argument\\n\")"
            )
        );
        assert!(!source.contains("simp at consumerStage consumer argument\\n          omega"));
    }

    #[test]
    fn module_roots_and_dependency_order_are_fail_closed() {
        assert!(validate_module_name("MxxGenerated.Test_Program").is_ok());
        assert!(validate_module_name("MxxGenerated/Program").is_err());
        assert!(validate_module_name("MxxGenerated..Program").is_err());

        let data = rendered_module("Root.Data".to_owned(), "import MxxIrCore.Program\n".to_owned());
        let certificate =
            rendered_module("Root.Certificate".to_owned(), "import Root.Data\n".to_owned());
        assert!(validate_module_order("Root", &[data.clone(), certificate.clone()]).is_ok());
        assert!(validate_module_order("Root", &[certificate, data]).is_err());
    }

    #[test]
    fn node_certificate_chunks_are_deterministic_and_never_split_scopes() {
        let scopes = vec![
            ("scope0".to_owned(), "scope0-BEGIN-aaaa-scope0-END\n".to_owned()),
            ("scope1".to_owned(), "scope1-BEGIN-bbbb-scope1-END\n".to_owned()),
            ("scope2".to_owned(), "scope2-BEGIN-cccc-scope2-END\n".to_owned()),
        ];
        let first = chunk_scope_sources(scopes.clone(), 50);
        let second = chunk_scope_sources(scopes, 50);
        assert_eq!(first, second);
        for (index, marker) in ["scope0", "scope1", "scope2"].iter().enumerate() {
            assert_eq!(first.iter().filter(|chunk| chunk.contains(marker)).count(), 1, "{index}");
        }
    }

    fn compile_generated_modules(rendered: &RenderedLeanProgram, tag: &str) {
        let root = std::env::temp_dir()
            .join(format!("mxx-ir-equation-fixture-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let lean_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lean");
        let lean = std::process::Command::new("lake")
            .args(["env", "which", "lean"])
            .current_dir(&lean_dir)
            .output()
            .unwrap();
        assert!(lean.status.success());
        let lean = String::from_utf8(lean.stdout).unwrap();
        let lean = lean.trim();
        let path_env = std::process::Command::new("lake")
            .args(["env", "printenv", "LEAN_PATH"])
            .current_dir(&lean_dir)
            .output()
            .unwrap();
        assert!(path_env.status.success());
        let path_env = String::from_utf8(path_env.stdout).unwrap();
        for module in &rendered.modules {
            let path = root.join(&module.relative_path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &module.source).unwrap();
            let object = path.with_extension("olean");
            let check = std::process::Command::new(lean)
                .args([
                    "--root",
                    root.to_str().unwrap(),
                    "-o",
                    object.to_str().unwrap(),
                    path.to_str().unwrap(),
                ])
                .env("LEAN_PATH", format!("{}:{}", root.display(), path_env.trim()))
                .current_dir(&lean_dir)
                .output()
                .unwrap();
            assert!(
                check.status.success(),
                "generated module {} failed to compile:\n{}\n{}",
                module.module_name,
                String::from_utf8_lossy(&check.stdout),
                String::from_utf8_lossy(&check.stderr)
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn primitive_equations_are_emitted_from_concrete_nodes() {
        let value =
            NodeHandle::new(NodeKind::ConstantInt(7.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let matrix_type = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let matrix = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: matrix_type.clone(),
                value: ConstantMatrix::Gadget { base: IntExpr::constant(2), small: false },
            },
            vec![],
            vec![WireType::Matrix(matrix_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "primitive-equation",
            Vec::new(),
            std::collections::BTreeMap::from([
                ("out".to_owned(), GraphOutput { value, confidentiality: None }),
                ("matrix".to_owned(), GraphOutput { value: matrix, confidentiality: None }),
            ]),
            Vec::new(),
            Vec::new(),
            std::collections::BTreeMap::new(),
        )
        .unwrap();
        let validated = validate(&graph, &ParamEnv::default()).unwrap();
        let production = ProductionId {
            spec_hash: crate::encoding::spec_hash(&validated.source, &validated.bindings).unwrap(),
            execution_nonce: [4; 32],
        };
        let manifest = export_validated_manifest(production.clone(), &validated).unwrap();
        let linked = ValidatedLinkedProgram::new(vec![LinkedProgramStage::new(
            production, validated, manifest,
        )])
        .unwrap();
        let rendered = render_lean_program(&linked, "MxxGenerated.EquationFixture").unwrap();
        let equations = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("NodeEquations00"))
            .expect("primitive fixture must have an equation module");
        assert!(equations.source.contains("import MxxIrCore.ScopeInvariant"));
        assert!(equations.source.contains("generatedPrimitiveNodeEquation"));
        assert!(equations.source.contains("PrimitiveNodePayload.constantInt 7"));
        assert!(equations.source.contains("PrimitiveNodePayload.constantMatrix"));
        let roots = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("StageRoots00"))
            .expect("primitive fixture must have stage-root bridges");
        assert!(roots.source.contains("stage0_scope0_node1FromPublicEval"));
        assert!(roots.source.contains("stage0_scope0_node1ReachedPrimitiveRunFromPublicEval"));

        let root =
            std::env::temp_dir().join(format!("mxx-ir-equation-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let lean_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lean");
        for module in &rendered.modules {
            let path = root.join(&module.relative_path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &module.source).unwrap();
            let object = path.with_extension("olean");
            let lean = std::process::Command::new("lake")
                .args(["env", "which", "lean"])
                .current_dir(&lean_dir)
                .output()
                .unwrap();
            assert!(lean.status.success());
            let lean = String::from_utf8(lean.stdout).unwrap();
            let lean = lean.trim();
            let path_env = std::process::Command::new("lake")
                .args(["env", "printenv", "LEAN_PATH"])
                .current_dir(&lean_dir)
                .output()
                .unwrap();
            assert!(path_env.status.success());
            let path_env = String::from_utf8(path_env.stdout).unwrap();
            let check = std::process::Command::new(lean)
                .args([
                    "--root",
                    root.to_str().unwrap(),
                    "-o",
                    object.to_str().unwrap(),
                    path.to_str().unwrap(),
                ])
                .env("LEAN_PATH", format!("{}:{}", root.display(), path_env.trim()))
                .current_dir(&lean_dir)
                .output()
                .unwrap();
            assert!(
                check.status.success(),
                "generated module {} failed to compile:\n{}\n{}",
                module.module_name,
                String::from_utf8_lossy(&check.stdout),
                String::from_utf8_lossy(&check.stderr)
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsupported_primitives_do_not_receive_non_root_prefix_callbacks() {
        use crate::{
            linked::{ConcreteScope, ConcreteStructuralIntExpr},
            node::PreimageBinaryOp,
            types::ConcreteMatrixType,
        };

        let one = ConcreteStructuralIntExpr::Literal(1.into());
        let matrix_type = ConcreteMatrixType::scalar(17.into(), 1);
        let cases = vec![
            (
                ConcreteNodePayload::ConstantMatrix {
                    matrix_type: matrix_type.clone(),
                    value: ConcreteMatrixLiteral::Gadget { base: one.clone(), small: true },
                },
                "constant small gadget matrix",
            ),
            (
                ConcreteNodePayload::GadgetDecompose {
                    base: one.clone(),
                    small: true,
                    digit_count: one.clone(),
                },
                "small gadget decomposition",
            ),
            (
                ConcreteNodePayload::MatrixMulAccumulate {
                    coefficients: vec![one.clone()],
                    has_bias: false,
                },
                "matrixMulAccumulate",
            ),
            (ConcreteNodePayload::Tensor, "tensor"),
            (ConcreteNodePayload::PreimageBinary(PreimageBinaryOp::Add), "preimageBinary"),
            (ConcreteNodePayload::PreimageConcatColumns, "preimageConcatColumns"),
            (
                ConcreteNodePayload::DecompositionEntry { row: one.clone(), column: one.clone() },
                "decompositionEntry",
            ),
            (
                ConcreteNodePayload::LiftIntegerToConstantPolynomial {
                    matrix_type: matrix_type.clone(),
                },
                "liftIntegerToConstantPolynomial",
            ),
            (
                ConcreteNodePayload::ThresholdDecode {
                    plaintext_modulus: one.clone(),
                    length: one.clone(),
                    output_bool: true,
                },
                "thresholdDecode",
            ),
            (
                ConcreteNodePayload::CrtRecompose {
                    plaintext_moduli: vec![one.clone()],
                    reconstruction_coefficients: vec![one.clone()],
                },
                "crtRecompose",
            ),
            (
                ConcreteNodePayload::PackPolynomialCoefficients {
                    matrix_type,
                    coefficient_bits: one,
                },
                "packPolynomialCoefficients",
            ),
        ];

        for (kind, constructor) in cases {
            assert!(
                render_primitive_payload_proof(&kind).unwrap().is_none(),
                "{constructor} must remain classified as unsupported"
            );
            let scope = ConcreteScope {
                id: 7,
                structural_slots: vec![],
                nodes: vec![ConcreteNode {
                    kind,
                    arguments: vec![],
                    outputs: vec![],
                    child_scope: None,
                }],
                inputs: vec![],
                outputs: vec![],
            };
            assert!(
                render_flat_scope_prefix_callback(3, &scope, 1, false).unwrap().is_none(),
                "{constructor} must not receive a successful prefix callback"
            );
        }
    }

    #[test]
    fn gadget_equation_exposes_backend_certificate_hook() {
        let matrix_type = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let input = NodeHandle::new(
            NodeKind::Input {
                name: "target".to_owned(),
                wire_type: WireType::Matrix(matrix_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(matrix_type.clone())],
        )
        .output(0)
        .unwrap();
        let decomposition = NodeHandle::new(
            NodeKind::GadgetDecompose {
                base: IntExpr::constant(2),
                small: false,
                digit_count: IntExpr::constant(1),
            },
            vec![input],
            vec![WireType::Preimage(matrix_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "gadget-equation",
            Vec::new(),
            std::collections::BTreeMap::from([(
                "out".to_owned(),
                GraphOutput { value: decomposition, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            std::collections::BTreeMap::new(),
        )
        .unwrap();
        let validated = validate(&graph, &ParamEnv::default()).unwrap();
        let production = ProductionId {
            spec_hash: crate::encoding::spec_hash(&validated.source, &validated.bindings).unwrap(),
            execution_nonce: [11; 32],
        };
        let manifest = export_validated_manifest(production.clone(), &validated).unwrap();
        let linked = ValidatedLinkedProgram::new(vec![LinkedProgramStage::new(
            production, validated, manifest,
        )])
        .unwrap();
        let rendered = render_lean_program(&linked, "MxxGenerated.GadgetEquationFixture").unwrap();
        let equations = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("NodeEquations00"))
            .expect("gadget fixture must have an equation module");
        assert!(!equations.source.contains("GadgetCertificate"));
        let roots = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("StageRoots00"))
            .expect("gadget fixture must have a stage-root equation module");
        assert!(roots.source.contains("GadgetExecution"));
        assert!(roots.source.contains("FromPublicEval"));
        assert!(roots.source.contains("evalPrimitiveNode_gadgetDecompose_success"));
        assert!(roots.source.contains("Mxx.IR.eval backend program env"));
        assert!(!roots.source.contains("gadgetDecompose_node_certificate"));
        compile_generated_modules(&rendered, "gadget");
    }

    #[test]
    fn sampler_equations_are_emitted_and_kernel_checked() {
        let matrix_type = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let sampled = NodeHandle::new(
            NodeKind::UniformResidueSample { matrix_type: matrix_type.clone() },
            Vec::new(),
            vec![WireType::Matrix(matrix_type.clone())],
        )
        .output(0)
        .unwrap();
        let value = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            vec![sampled.clone(), sampled],
            vec![WireType::Matrix(matrix_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "sampler-equation",
            Vec::new(),
            std::collections::BTreeMap::from([(
                "out".to_owned(),
                GraphOutput { value, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            std::collections::BTreeMap::new(),
        )
        .unwrap();
        let validated = validate(&graph, &ParamEnv::default()).unwrap();
        let production = ProductionId {
            spec_hash: crate::encoding::spec_hash(&validated.source, &validated.bindings).unwrap(),
            execution_nonce: [8; 32],
        };
        let manifest = export_validated_manifest(production.clone(), &validated).unwrap();
        let linked = ValidatedLinkedProgram::new(vec![LinkedProgramStage::new(
            production, validated, manifest,
        )])
        .unwrap();
        let rendered = render_lean_program(&linked, "MxxGenerated.SamplerEquationFixture").unwrap();
        let equations = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("NodeEquations00"))
            .expect("sampler fixture must have an equation module");
        assert!(equations.source.contains("evalScope_success_sampler_step"));
        assert!(equations.source.contains("generatedPrimitiveNodeEquation"));
        let roots = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("StageRoots00"))
            .expect("sampler fixture must have stage-root bridges");
        assert!(roots.source.contains("NodeResult.ofValues sampled"));
        compile_generated_modules(&rendered, "sampler");
    }

    #[test]
    fn artifact_equations_are_emitted_and_kernel_checked() {
        use crate::artifact::ArtifactConfidentiality;
        let matrix_type = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let make_graph = |name: &str, artifact: Option<crate::node::ArtifactInput>| {
            let input = NodeHandle::new(
                NodeKind::Input {
                    name: "input".to_owned(),
                    wire_type: WireType::Matrix(matrix_type.clone()),
                    artifact,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix_type.clone())],
            )
            .output(0)
            .unwrap();
            Graph::freeze(
                name,
                Vec::new(),
                std::collections::BTreeMap::from([(
                    "out".to_owned(),
                    GraphOutput {
                        value: input,
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                )]),
                Vec::new(),
                Vec::new(),
                std::collections::BTreeMap::new(),
            )
            .unwrap()
            .0
        };
        let producer_graph =
            validate(&make_graph("artifact-producer", None), &ParamEnv::default()).unwrap();
        let producer_id = ProductionId {
            spec_hash: crate::encoding::spec_hash(&producer_graph.source, &producer_graph.bindings)
                .unwrap(),
            execution_nonce: [9; 32],
        };
        let producer_manifest =
            export_validated_manifest(producer_id.clone(), &producer_graph).unwrap();
        let consumer_input = crate::node::ArtifactInput {
            production_id: producer_id.clone(),
            artifact_name: "out".to_owned(),
            confidentiality: ArtifactConfidentiality::Public,
        };
        let consumer_graph = validate_with_manifests(
            &make_graph("artifact-consumer", Some(consumer_input)),
            &ParamEnv::default(),
            &std::collections::BTreeMap::from([(producer_id.clone(), producer_manifest.clone())]),
        )
        .unwrap();
        let consumer_id = ProductionId {
            spec_hash: crate::encoding::spec_hash(&consumer_graph.source, &consumer_graph.bindings)
                .unwrap(),
            execution_nonce: [10; 32],
        };
        let consumer_manifest =
            export_validated_manifest(consumer_id.clone(), &consumer_graph).unwrap();
        let linked = ValidatedLinkedProgram::new(vec![
            LinkedProgramStage::new(producer_id, producer_graph, producer_manifest),
            LinkedProgramStage::new(consumer_id, consumer_graph, consumer_manifest),
        ])
        .unwrap();
        let rendered =
            render_lean_program(&linked, "MxxGenerated.ArtifactEquationFixture").unwrap();
        let equations = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("NodeEquations00"))
            .expect("artifact fixture must have an equation module");
        assert!(equations.source.contains("evalScope_success_artifact_step"));
        assert!(!equations.source.contains("NodeResult.ofValues #[value] ="));
        assert!(rendered.modules.iter().any(|module| {
            module.source.contains("ArtifactValueFromPublicEval") &&
                module.source.contains("lookup producerScope.values link.producer = some value") &&
                module.source.contains("traceValueAt finalTrace")
        }));
        compile_generated_modules(&rendered, "artifact");
    }

    #[test]
    fn structural_equation_wrappers_are_emitted_for_grid_and_loop_nodes() {
        let family_type = WireType::Family {
            element: Box::new(WireType::Int),
            shape: vec![IntExpr::constant(1)],
        };
        let family_input = NodeHandle::new(
            NodeKind::Input { name: "scalar".to_owned(), wire_type: WireType::Int, artifact: None },
            Vec::new(),
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let grid_body = with_new_construction_scope(|scope| {
            let element = NodeHandle::new(
                NodeKind::Input {
                    name: "element".to_owned(),
                    wire_type: WireType::Int,
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("grid-equation-body", scope, vec![element.clone()], vec![element])
                .unwrap()
        });
        let grid = NodeHandle::parallel_grid(
            grid_body,
            vec![family_input.clone()],
            vec![family_type.clone()],
            ParallelGrid {
                shape: vec![IntExpr::constant(1)],
                index_slots: vec![0],
                bindings: Vec::new(),
                input_modes: vec![GridInputMode::Broadcast],
            },
        )
        .output(0)
        .unwrap();
        let loop_body = with_new_construction_scope(|scope| {
            let carried = NodeHandle::new(
                NodeKind::Input {
                    name: "carried".to_owned(),
                    wire_type: WireType::Int,
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            let one = NodeHandle::new(
                NodeKind::ConstantInt(1.into()),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            let incremented = NodeHandle::new(
                NodeKind::IntBinary(crate::node::IntBinaryOp::Add),
                vec![carried.clone(), one],
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("loop-equation-body", scope, vec![carried], vec![incremented])
                .unwrap()
        });
        let initial = NodeHandle::new(
            NodeKind::Input {
                name: "initial".to_owned(),
                wire_type: WireType::Int,
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let loop_node = NodeHandle::sequential_loop(
            loop_body,
            vec![initial],
            vec![WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(1),
                index_slot: 1,
                bindings: Vec::new(),
                carried_count: 1,
            },
        )
        .output(0)
        .unwrap();
        let zero =
            NodeHandle::new(NodeKind::ConstantInt(0.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let loop_output = NodeHandle::new(
            NodeKind::IntBinary(crate::node::IntBinaryOp::Add),
            vec![loop_node, zero],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let selected_grid_value = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices: vec![IndexExpr::Constant(0.into())] },
            vec![grid],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let combined_output = NodeHandle::new(
            NodeKind::IntBinary(crate::node::IntBinaryOp::Add),
            vec![loop_output, selected_grid_value.clone()],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "structural-equations",
            Vec::new(),
            std::collections::BTreeMap::from([
                (
                    "grid".to_owned(),
                    GraphOutput { value: selected_grid_value, confidentiality: None },
                ),
                ("loop".to_owned(), GraphOutput { value: combined_output, confidentiality: None }),
            ]),
            Vec::new(),
            Vec::new(),
            std::collections::BTreeMap::new(),
        )
        .unwrap();
        let validated = validate(&graph, &ParamEnv::default()).unwrap();
        let production = ProductionId {
            spec_hash: crate::encoding::spec_hash(&validated.source, &validated.bindings).unwrap(),
            execution_nonce: [5; 32],
        };
        let manifest = export_validated_manifest(production.clone(), &validated).unwrap();
        let linked = ValidatedLinkedProgram::new(vec![LinkedProgramStage::new(
            production, validated, manifest,
        )])
        .unwrap();
        let rendered =
            render_lean_program(&linked, "MxxGenerated.StructuralEquationFixture").unwrap();
        let equations = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("NodeEquations00"))
            .expect("structural fixture must have an equation module");
        assert!(equations.source.contains("generatedParallelGridNodeEquation"));
        assert!(equations.source.contains("generatedSequentialLoopNodeEquation"));
        assert!(equations.source.contains("evalScope_success_input_step"));
        assert!(equations.source.contains("ParallelGridEquation"));
        assert!(equations.source.contains("SequentialLoopEquation"));
        assert!(equations.source.contains("PrimitiveNodePayload.intBinary .add"));
        assert!(equations.source.contains("PrefixSteps"));
        assert!(equations.source.contains("ScopeFreePrefixSteps"));
        assert!(equations.source.contains("Mxx.IR.ScopeFreeStep"));
        assert!(equations.source.contains("SuffixAvoids"));
        assert!(equations.source.contains("Mxx.IR.AvoidingScopeStep"));
        assert!(equations.source.contains("inputs path index values fuel"));
        compile_generated_modules(&rendered, "structural");
    }

    #[test]
    fn multi_stage_root_bridge_is_generated_and_kernel_checked() {
        let make_stage = |name: &str, nonce: u8| {
            let value = NodeHandle::new(
                NodeKind::ConstantInt(7.into()),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            let (graph, _) = Graph::freeze(
                name,
                Vec::new(),
                std::collections::BTreeMap::from([(
                    "out".to_owned(),
                    GraphOutput { value, confidentiality: None },
                )]),
                Vec::new(),
                Vec::new(),
                std::collections::BTreeMap::new(),
            )
            .unwrap();
            let validated = validate(&graph, &ParamEnv::default()).unwrap();
            let production = ProductionId {
                spec_hash: crate::encoding::spec_hash(&validated.source, &validated.bindings)
                    .unwrap(),
                execution_nonce: [nonce; 32],
            };
            let manifest = export_validated_manifest(production.clone(), &validated).unwrap();
            LinkedProgramStage::new(production, validated, manifest)
        };
        let linked = ValidatedLinkedProgram::new(vec![
            make_stage("stage-zero", 6),
            make_stage("stage-one", 7),
        ])
        .unwrap();
        let rendered =
            render_lean_program(&linked, "MxxGenerated.MultiStageEquationFixture").unwrap();
        let roots = rendered
            .modules
            .iter()
            .find(|module| module.module_name.ends_with("StageRoots00"))
            .expect("multi-stage fixture must have root equations");
        assert!(roots.source.contains("stage0RootSuccess"));
        assert!(roots.source.contains("stage1RootSuccess"));
        compile_generated_modules(&rendered, "multi-stage");
    }
}
