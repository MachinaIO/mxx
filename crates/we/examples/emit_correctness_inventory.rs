use mxx_gadgets::circuit::BooleanCircuitShape;
use mxx_ir_core::{
    Graph, ParamEnv, RealExpr,
    artifact::{ProductionId, SpecHash, export_validated_manifest},
    encoding::spec_hash,
    inventory::inventory,
    lean::{ExportOptions, export},
    node::NodeKind,
    validate, validate_with_manifests,
};
use mxx_we::diamond::{DiamondWeCompiler, DiamondWeConfig};
use std::{collections::BTreeMap, error::Error};

fn node_kind_name(kind: &NodeKind) -> String {
    let debug = format!("{kind:?}");
    debug.split(['(', '{', ' ']).next().unwrap_or("<unknown>").to_owned()
}

fn hex_digest(bytes: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn root_input_names(graph: &Graph) -> Vec<String> {
    graph
        .root_scope()
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, artifact: None, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn print_inventory(
    label: &str,
    graph: &Graph,
    bindings: &ParamEnv,
    manifests: Option<&BTreeMap<ProductionId, mxx_ir_core::artifact::Manifest>>,
) -> Result<(), Box<dyn Error>> {
    let snapshot = inventory(graph)?;
    println!(
        "graph {label} name={} digest={}",
        snapshot.graph_name, snapshot.symbolic_graph_digest
    );
    println!(
        "  root_outputs={} effects={} scopes={} structural_edges={}",
        snapshot.root_outputs.len(),
        snapshot.effect_roots.len(),
        snapshot.scopes.len(),
        snapshot.structural_edges.len()
    );
    for scope in &snapshot.scopes {
        println!(
            "  scope {:?} inputs={} outputs={} nodes={}",
            scope.scope,
            scope.inputs.len(),
            scope.outputs.len(),
            scope.nodes.len()
        );
        let mut kinds = BTreeMap::<String, usize>::new();
        for node in &scope.nodes {
            let name = node_kind_name(&node.kind);
            *kinds.entry(name).or_default() += 1;
        }
        for (kind, count) in kinds {
            println!("    op_kind {kind} count={count}");
        }
    }
    let validated = match manifests {
        Some(manifests) => validate_with_manifests(graph, bindings, manifests),
        None => validate(graph, bindings),
    };
    match validated {
        Ok(validated) => match export(&validated, &ExportOptions::default()) {
            Ok(artifact) => println!(
                "  generic_export=success module={} visits={} digest={}",
                artifact.module_name,
                artifact.static_node_visits,
                hex_digest(artifact.digest)
            ),
            Err(error) => println!("  generic_export=error {error}"),
        },
        Err(error) => println!("  validation=error {error}"),
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
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
            bgg_tag: b"correctness-inventory".to_vec(),
        },
        BooleanCircuitShape { instance_width: 1, witness_width: 1, depth: 2, max_layer_width: 3 },
    )?;
    let protocol = compiler.protocol_decl()?;
    let declaration = protocol.protocol();
    let bindings = compiler.circuit_bindings()?;
    let encryption_stage = declaration
        .stages()
        .iter()
        .find(|stage| stage.id.0 == "encrypt")
        .ok_or("missing encrypt stage")?;
    let validated_encryption = validate(&encryption_stage.graph, &bindings)?;
    let placeholder_production =
        ProductionId { spec_hash: SpecHash([0; 32]), execution_nonce: [0; 32] };
    let manifest =
        export_validated_manifest(placeholder_production.clone(), &validated_encryption)?;
    let manifests = BTreeMap::from([(placeholder_production.clone(), manifest)]);
    let concrete_hash = spec_hash(&validated_encryption.source, &validated_encryption.bindings)?;

    println!(
        "configuration=structural_only modulus=257 ring_dimension=8 gadget_base=4 digit_count=2"
    );
    println!(
        "configuration_note=inventory wiring check only; no concrete CRT capacity or candidate acceptance"
    );
    println!(
        "workflow_manifest=derived from validated encrypt graph; placeholder_production={:?} concrete_encrypt_spec_hash={:?}",
        placeholder_production, concrete_hash
    );
    println!("protocol parameters={}", declaration.params.len());
    for parameter in &declaration.params {
        println!("  parameter {} kind={:?}", parameter.name, parameter.kind);
    }
    for stage in &declaration.bundle.workflow.stages {
        println!("stage {} bindings={}", stage.id.0, stage.bindings.len());
        for binding in &stage.bindings {
            println!(
                "  artifact_binding consumer={} producer_stage={} producer_output={}",
                binding.consumer_input.0, binding.producer_stage.0, binding.producer_output.0
            );
        }
        let stage_manifests = (stage.id.0 == "decrypt").then_some(&manifests);
        print_inventory(
            &format!("stage:{}", stage.id.0),
            &stage.graph,
            &bindings,
            stage_manifests,
        )?;
    }
    for binding in &declaration.bundle.input_bindings {
        println!(
            "protocol_input_binding input={} destinations={:?}",
            binding.input.0, binding.destinations
        );
    }
    for (index, requirement) in declaration.bundle.requirements.iter().enumerate() {
        let inputs = root_input_names(&requirement.graph);
        let kind = if inputs.is_empty() {
            "valid-parameters"
        } else if inputs.iter().any(|name| name == "boolean-instance" || name == "boolean-witness")
        {
            "satisfaction"
        } else {
            "validity"
        };
        println!("requirement {index} root_inputs={:?} kind={}", inputs, kind);
        print_inventory(&format!("requirement:{index}"), &requirement.graph, &bindings, None)?;
    }
    println!("ideal root_inputs={:?}", root_input_names(&declaration.bundle.ideal.graph));
    print_inventory("ideal", &declaration.bundle.ideal.graph, &bindings, None)?;
    println!("endpoint_specs={:?}", declaration.bundle.endpoint_specs);
    for target in &declaration.bundle.operational_decoder_targets {
        println!(
            "decoder_target id={} residual={}:{} decoder={}:{} kind={:?}",
            target.target_id,
            target.residual_stage.0,
            target.residual_output,
            target.decoder_stage.0,
            target.decoder_node.0,
            target.kind
        );
    }
    Ok(())
}
