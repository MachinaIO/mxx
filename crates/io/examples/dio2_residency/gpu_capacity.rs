//! Diagnostic physical planning with synthetic ordinary inputs, without artifact execution.
//! This uses the real graph and backend allocation ledger; it is not protocol acceptance.
use mxx_backends::{
    GpuRuntime, RuntimeValue,
    backend::poly_gpu::gpu_backend_on,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
    },
};
use mxx_io::dio2::{
    Dio2Parameters,
    scheme::{INITIAL_STATE_INPUT, LABELS_INPUT, STEP_DATA_INPUT, X_INPUT, Y_INPUT},
};
use mxx_ir_core::{
    ValidatedGraph,
    node::NodeKind,
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use std::collections::BTreeMap;

pub fn plan(
    parameters: &Dio2Parameters,
    graph: &ValidatedGraph,
    inputs: &BTreeMap<String, RuntimeValue>,
) -> Result<(), String> {
    let mut inputs = inputs.clone();
    let mut runtime = GpuRuntime::new(gpu_backend_on(
        parameters.rings().map_err(|e| e.to_string())?.into_iter().map(|ring| {
            GpuDCRTPolyParams::new(ring.ring_dimension(), ring.to_crt().0, ring.base_bits(), None)
        }),
        detected_gpu_device_ids(),
    ))
    .map_err(|e| e.to_string())?;
    // The example has one-bit state/chunks and Boolean step tables. Labels range
    // over the full label modulus, not just bits. Explicit ranges avoid
    // specializing the plan to the chosen placeholder pattern. Selector/table
    // artifacts remain deferred; this does not create their enormous payloads.
    for (index, node) in graph.source.root_scope().nodes().iter().enumerate() {
        let NodeKind::Input { name, artifact: None, .. } = node.kind() else { continue };
        let ty =
            &graph.root_scope().wire_types[&WireRef { node: NodeId(index as u64), port: Port(0) }];
        match ty {
            ConcreteWireType::IndexedFamily { element, count }
                if element.as_ref() == &ConcreteWireType::Int =>
            {
                let count = usize::try_from(*count)
                    .map_err(|_| format!("ordinary input {name} is not host addressable"))?;
                let maximum = match name.as_str() {
                    LABELS_INPUT => parameters.label_modulus() - 1,
                    STEP_DATA_INPUT | X_INPUT | Y_INPUT => 1,
                    _ => return Err(format!("unsupported ordinary integer family {name}")),
                };
                inputs.entry(name.clone()).or_insert_with(|| {
                    RuntimeValue::integer_values(
                        (0..count)
                            .map(|i| if i % 2 == 0 { 0.into() } else { maximum.into() })
                            .collect(),
                    )
                });
                runtime
                    .options_mut()
                    .integer_input_ranges
                    .insert(name.clone(), 0.into()..=maximum.into());
            }
            ConcreteWireType::Int if name == INITIAL_STATE_INPUT => {
                inputs.entry(name.clone()).or_insert_with(|| RuntimeValue::Int(0.into()));
                runtime
                    .options_mut()
                    .integer_input_ranges
                    .insert(name.clone(), 0.into()..=1.into());
            }
            ConcreteWireType::Bytes { length: 32 } => {
                inputs.entry(name.clone()).or_insert_with(|| {
                    RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec().into())
                });
            }
            other if !inputs.contains_key(name) => {
                return Err(format!("unsupported unbound capacity input {name}: {other:?}"))
            }
            _ => {}
        }
    }
    // PRF selector artifacts contain Boolean slot selectors; public tables contain
    // LWR coordinate indices (including the padded zero coordinate). These are
    // byte-capacity bounds, not fabricated payloads or a protocol execution.
    for (scope_id, scope) in &graph.scopes {
        let source = graph.source.scope(scope_id).expect("validated scope");
        for (wire, descriptor) in &scope.artifact_inputs {
            if descriptor.artifact_type != mxx_ir_core::artifact::ArtifactType::Int {
                continue;
            }
            let node = source.node(wire.node).expect("artifact input");
            let NodeKind::Input { artifact: Some(artifact), .. } = node.kind() else { continue };
            let bytes = if artifact.artifact_name.ends_with("/selectors") {
                1
            } else if artifact.artifact_name.ends_with("/table") {
                num_bigint::BigInt::from(parameters.prf.lwr_dimension).to_signed_bytes_le().len()
            } else {
                return Err(format!("unsupported integer artifact {}", artifact.artifact_name));
            };
            runtime.options_mut().artifact_payload_size_bounds.insert(
                mxx_backends::ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: None,
                },
                bytes,
            );
        }
    }
    let plan = runtime.plan(graph.clone(), &inputs).map_err(|e| e.to_string())?;
    println!(
        "{} synthetic_input_physical_plan W_limit={} planned_device_bytes={:?} regions={} launches={}",
        graph.source.name(),
        runtime.options().max_parallel_instances,
        plan.planned_device_bytes().map_err(|e| e.to_string())?,
        plan.compiled_region_count(),
        plan.compiled_launch_count()
    );
    Ok(())
}
