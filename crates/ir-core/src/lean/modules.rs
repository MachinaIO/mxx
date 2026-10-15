//! Declaration-boundary module partitioning for large generated Lean artifacts.
//!
//! Every declaration retains its namespace and definition. Parts import their predecessor;
//! the original module re-exports the last part. Local notations and section environments are
//! restored at part boundaries without repeating definitions or changing propositions.

/// The maximum source bytes of a generated graph or facts module part.
pub(super) const MODULE_BYTES: usize = 3_000_000;

pub(super) fn partition(
    name: &str,
    source: &str,
    limit: usize,
) -> Result<Vec<(String, String)>, String> {
    if source.len() <= limit {
        return Ok(vec![(name.into(), source.into())]);
    }
    let namespace_line = source
        .lines()
        .find(|line| line.starts_with("namespace "))
        .ok_or("generated module has no namespace")?;
    let namespace = namespace_line.trim_start_matches("namespace ");
    let namespace_end = source.find(namespace_line).unwrap() + namespace_line.len() + 1;
    let footer = format!("\nend {namespace}\n");
    let body_end = source.rfind(footer.as_str()).ok_or("generated module has no namespace end")?;
    let mut body_start = namespace_end;
    for line in source[namespace_end..body_end].split_inclusive('\n') {
        if line.starts_with("local notation \"mxx") {
            body_start += line.len();
        } else if body_start > namespace_end {
            break;
        } else {
            body_start += line.len();
            if line.starts_with("abbrev ") ||
                line.starts_with("structure ") ||
                line.starts_with("def ")
            {
                body_start -= line.len();
                break;
            }
        }
    }
    if let Some(position) = source[namespace_end..body_end].rfind("\nlocal notation \"mxx") {
        let start = namespace_end + position + 1;
        body_start = start + source[start..].find('\n').ok_or("unterminated notation")? + 1;
    }
    let prelude = &source[..body_start];
    let notations = prelude
        .lines()
        .filter(|line| line.starts_with("local notation "))
        .map(|line| format!("{line}\n"))
        .collect::<String>();
    let common = format!("{}\n{notations}\n", &source[..namespace_end]);
    let mut parts = Vec::new();
    let mut current = prelude.to_string();
    let mut section: Option<&str> = None;
    let mut has_declaration = false;
    for block in source[body_start..body_end].split("\n\n").filter(|block| !block.trim().is_empty())
    {
        let closing = if section.is_some() { "\nend\n" } else { "" };
        if has_declaration && current.len() + block.len() + closing.len() + footer.len() + 2 > limit
        {
            current.push_str(closing);
            current.push_str(&footer);
            let part = format!("{name}_Part_{}", parts.len());
            parts.push((part.clone(), current));
            current = format!("import {part}\n{common}");
            if let Some(environment) = section {
                current.push_str(environment);
                current.push_str("\n\n");
            }
        }
        current.push_str(block);
        current.push_str("\n\n");
        has_declaration = true;
        if block.trim_start().starts_with("section\n") ||
            block.trim_start().starts_with("noncomputable section\n")
        {
            if section.is_some() {
                return Err("nested generated sections cannot be partitioned".into());
            }
            section = Some(block);
        } else if block.trim() == "end" {
            section = None;
        }
        let closing = if section.is_some() { "\nend\n" } else { "" };
        if current.len() + closing.len() + footer.len() > limit {
            return Err(format!("a declaration in {name} exceeds the {limit}-byte module limit"));
        }
    }
    if section.is_some() {
        return Err("unclosed generated section".into());
    }
    current.push_str(&footer);
    let last = format!("{name}_Part_{}", parts.len());
    parts.push((last.clone(), current));
    parts.push((name.into(), format!("import {last}\n")));
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_restores_local_notations_and_section_environments() {
        let mut source = "namespace Partitioned\n\nstructure Params where\n  value : Nat\n\nlocal notation \"mxxNat0\" => 7\n\nnoncomputable section\nvariable {params : Params}\nlocal notation \"ctx\" => params.value\n\n".to_string();
        for index in 0..30 {
            source.push_str(&format!("def value_{index} : Nat := ctx + mxxNat0\n\n"));
        }
        source.push_str("end\n\nend Partitioned\n");
        let modules = partition("Partitioned", &source, 700).unwrap();
        assert!(modules.len() > 3);
        assert!(modules.iter().all(|(_, source)| source.len() <= 700));
        for index in 0..30 {
            assert_eq!(
                modules
                    .iter()
                    .filter(|(_, source)| source.contains(&format!("def value_{index} :")))
                    .count(),
                1
            );
        }
        assert_eq!(
            modules.iter().filter(|(_, source)| source.contains("structure Params")).count(),
            1
        );
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/module_partition");
        std::fs::create_dir_all(&directory).unwrap();
        for (name, source) in modules {
            std::fs::write(directory.join(format!("{name}.lean")), source).unwrap();
        }
    }

    #[test]
    fn test_partition_preserves_generated_graph_and_facts_declarations() {
        use crate::{
            NodeHandle,
            graph::{Graph, GraphOutput},
            node::{IntBinaryOp, NodeKind},
            types::WireType,
        };
        use std::collections::BTreeMap;
        let constant =
            NodeHandle::new(NodeKind::ConstantInt(1.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let mut value = constant.clone();
        for _ in 0..64 {
            value = NodeHandle::new(
                NodeKind::IntBinary(IntBinaryOp::Add),
                vec![value, constant.clone()],
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
        }
        let (graph, _) = Graph::freeze(
            "partition",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value, availability: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated =
            crate::ring::test_validate(&graph, &crate::expr::ParamEnv::default()).unwrap();
        let artifact = super::super::export(
            &validated,
            &super::super::ExportOptions {
                module_name: "PartitionedGraph".into(),
                namespace: "PartitionedGraph".into(),
                facts: true,
                ..Default::default()
            },
        )
        .unwrap();
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/generated_module_partition");
        std::fs::create_dir_all(&directory).unwrap();
        let mut total = 0;
        for (name, source) in [
            ("PartitionedGraph", artifact.source.as_str()),
            ("PartitionedGraph_Facts", artifact.facts_source.as_ref().unwrap().as_str()),
        ] {
            let modules = partition(name, source, 3_500).unwrap();
            total += modules.len();
            assert!(modules.iter().all(|(_, source)| source.len() <= 3_500));
            for (name, source) in modules {
                std::fs::write(directory.join(format!("{name}.lean")), source).unwrap();
            }
        }
        assert!(total > 4);
    }
}
