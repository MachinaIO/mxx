//! Warmup binding of scalar chains that consume device-produced values.
use super::{super::gpu_prepared_lowering::PreparedProgram, *};
use mxx_ir_core::{
    node::{IntBinaryOp, IntCompareOp, RealBinaryOp},
    types::ConcreteWireType,
};
use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedScalarOpcode;

pub(super) fn stage_runtime_scalar(
    buffer: &GpuPreparedScalarBuffer,
    value: &PreparedRuntimeValue,
) -> Result<(), String> {
    buffer.upload(|words, _| {
        words.fill(0);
        match value {
            PreparedRuntimeValue::Int(value) => {
                if value.bits() as usize >= words.len() * 64 {
                    return Err("prepared integer exceeds fixed signed capacity".into());
                }
                for (word, digit) in words.iter_mut().zip(value.magnitude().iter_u64_digits()) {
                    *word = digit;
                }
                if value.sign() == num_bigint::Sign::Minus {
                    let mut carry = true;
                    for word in words {
                        let (next, overflow) = (!*word).overflowing_add(u64::from(carry));
                        *word = next;
                        carry = overflow;
                    }
                }
            }
            PreparedRuntimeValue::Real(value) => words[0] = value.to_bits(),
            PreparedRuntimeValue::Bool(value) => words[0] = u64::from(*value),
            _ => return Err("prepared scalar staging input is not scalar".into()),
        }
        Ok(())
    })
}

pub(super) fn allocate_scalar_buffer(
    backend: &mut GpuDcrtBackend,
    region: &mut Arc<crate::gpu_memory::GpuMemoryRegion>,
    anchor: &Arc<GpuDCRTPolyMatrix>,
    device: i32,
    count: usize,
    words: usize,
) -> Result<Arc<GpuPreparedScalarBuffer>, String> {
    let bytes = (words + 1)
        .checked_mul(count)
        .and_then(|words| words.checked_mul(8))
        .ok_or("scalar capacity overflow")?;
    let claims = [
        GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::PinnedHost,
            bytes,
            alignment: 8,
        }),
        GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::BatchWorkspace,
            bytes,
            alignment: 8,
        }),
        GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::CompletionEvent,
            bytes: 0,
            alignment: 1,
        }),
        GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::CompletionEvent,
            bytes: 0,
            alignment: 1,
        }),
    ];
    bind_prepared_claims(backend, region, anchor.params(), device, &claims, || {
        GpuPreparedScalarBuffer::bind(Arc::clone(anchor), count, words)
    })
}

fn constant(
    kind: &NodeKind,
    environment: &mxx_ir_core::expr::ParamEnv,
) -> Result<Option<PreparedRuntimeValue>, String> {
    Ok(match kind {
        NodeKind::ConstantInt(value) => Some(PreparedRuntimeValue::Int(value.clone())),
        NodeKind::EvaluateInt(value) => Some(PreparedRuntimeValue::Int(
            value.evaluate(environment).map_err(|error| error.to_string())?,
        )),
        NodeKind::ConstantReal(value) => Some(PreparedRuntimeValue::Real(
            value.evaluate_f64(environment).map_err(|error| error.to_string())?,
        )),
        NodeKind::ConstantBool(value) => Some(PreparedRuntimeValue::Bool(*value)),
        _ => None,
    })
}

/// Evaluate only magnitude bounds at warmup. The same two-bank replay walks
/// these abstract values, so a loop's integer capacity reflects all iterations
/// without cloning its command tape or guessing a runtime allocation margin.
fn scalar_capacities(program: &PreparedProgram) -> Result<BTreeMap<usize, usize>, String> {
    use super::super::gpu_prepared_lowering::PreparedReplayStep;
    use num_bigint::BigUint;

    fn visit(
        program: &PreparedProgram,
        steps: &[PreparedReplayStep],
        parent: Option<usize>,
        bounds: &mut BTreeMap<usize, BigUint>,
        capacities: &mut BTreeMap<usize, usize>,
    ) -> Result<(), String> {
        for step in steps {
            match step {
                PreparedReplayStep::Node(id) => {
                    let (arguments, outputs) = &program.node_bindings[id];
                    if !outputs.iter().any(|wire| program.device_scalar_wires.contains(wire)) {
                        continue;
                    }
                    let source = &program.node_sources[id];
                    let kind = parent
                        .and_then(|index| source.variant_indices.get(index))
                        .and_then(|index| source.variants.get(*index))
                        .unwrap_or(&source.kind);
                    let operand = |index: usize| {
                        arguments
                            .get(index)
                            .and_then(|wire| program.scalar_slots.get(wire))
                            .and_then(|slot| bounds.get(slot))
                            .cloned()
                            .unwrap_or_default()
                    };
                    let magnitude = if program
                        .scalar_commands
                        .get(id)
                        .is_some_and(|command| command.instructions.is_empty()) &&
                        !arguments.is_empty()
                    {
                        operand(0)
                    } else if let Some(wire) =
                        outputs.first().filter(|wire| program.inputs.contains(wire))
                    {
                        let input = program
                            .runtime_input_wires
                            .iter()
                            .position(|input| input == wire)
                            .unwrap();
                        let words = program.scalar_input_max_words[input];
                        (BigUint::from(1u8) <<
                            words.checked_mul(64).ok_or("scalar capacity overflow")?) -
                            BigUint::from(1u8)
                    } else {
                        match kind {
                            NodeKind::ThresholdDecode {
                                plaintext_modulus,
                                output_bool: false,
                                ..
                            } => {
                                plaintext_modulus
                                    .evaluate(&source.environment)
                                    .map_err(|error| error.to_string())?
                                    .magnitude()
                                    .clone() -
                                    BigUint::from(1u8)
                            }
                            NodeKind::ConstantInt(value) => value.magnitude().clone(),
                            NodeKind::EvaluateInt(value) => value
                                .evaluate(&source.environment)
                                .map_err(|error| error.to_string())?
                                .magnitude()
                                .clone(),
                            NodeKind::IntBinary(operation) => match operation {
                                IntBinaryOp::Add | IntBinaryOp::Subtract => operand(0) + operand(1),
                                IntBinaryOp::Multiply => operand(0) * operand(1),
                                IntBinaryOp::Divide => operand(0),
                                IntBinaryOp::Remainder => operand(0).min(operand(1)),
                            },
                            NodeKind::FamilyGetStatic { .. } |
                            NodeKind::FamilyGetDynamic |
                            NodeKind::Select { .. } => {
                                let candidates = match program.selection_commands.get(id) {
                                    Some(super::super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { slot }) => vec![*slot],
                                    Some(super::super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { candidates, .. } | super::super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { candidates, .. }) => candidates.to_vec(),
                                    _ => Vec::new(),
                                };
                                candidates
                                    .iter()
                                    .filter_map(|slot| bounds.get(slot))
                                    .max()
                                    .cloned()
                                    .unwrap_or_default()
                            }
                            _ => BigUint::from(1u8),
                        }
                    };
                    for wire in outputs.iter() {
                        let Some(&slot) = program.scalar_slots.get(wire) else { continue };
                        let words = match program.wire_types[wire] {
                            ConcreteWireType::Int | ConcreteWireType::ConstantInt => {
                                magnitude.bits().div_ceil(64) as usize + 1
                            }
                            _ => 1,
                        };
                        capacities
                            .entry(slot)
                            .and_modify(|capacity| *capacity = (*capacity).max(words))
                            .or_insert(words);
                        bounds.insert(slot, magnitude.clone());
                    }
                }
                PreparedReplayStep::Subgraph { body } => {
                    visit(program, body, parent, bounds, capacities)?
                }
                PreparedReplayStep::Parallel { counts, waves } => {
                    let active =
                        parent.and_then(|index| counts.get(index)).copied().unwrap_or(usize::MAX);
                    for body in waves.iter().flat_map(|wave| wave.iter()).take(active) {
                        visit(program, std::slice::from_ref(body), parent, bounds, capacities)?;
                    }
                }
                PreparedReplayStep::Sequential { count, counts, offsets, banks, tail, .. } => {
                    let count =
                        parent.and_then(|index| counts.get(index)).copied().unwrap_or(*count);
                    let base = parent
                        .and_then(|index| offsets.get(index))
                        .copied()
                        .unwrap_or_else(|| parent.unwrap_or(0).saturating_mul(count));
                    for iteration in 0..count {
                        visit(
                            program,
                            &banks[iteration & 1],
                            Some(base + iteration),
                            bounds,
                            capacities,
                        )?;
                    }
                    if count % 2 == 1 {
                        visit(program, tail, Some(base + count - 1), bounds, capacities)?;
                    }
                }
            }
        }
        Ok(())
    }
    let mut capacities = BTreeMap::new();
    visit(program, &program.replay, None, &mut BTreeMap::new(), &mut capacities)?;
    Ok(capacities)
}

pub(super) fn prepare_scalar_commands(
    backend: &mut GpuDcrtBackend,
    region: &mut Arc<crate::gpu_memory::GpuMemoryRegion>,
    program: &PreparedProgram,
    runtime_inputs: &[PreparedRuntimeValue],
    anchor: &Arc<GpuDCRTPolyMatrix>,
    device: i32,
    values: &mut BTreeMap<WireRef, (Arc<GpuPreparedScalarBuffer>, usize)>,
    commands: &mut Vec<PreparedCommand>,
) -> Result<(), String> {
    let capacities = scalar_capacities(program)?;
    let unused_capacity = capacities.values().copied().max().unwrap_or(1);
    let mut slots = values
        .iter()
        .filter_map(|(wire, value)| {
            program.scalar_slots.get(wire).map(|slot| (*slot, value.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    for wire in &program.device_scalar_wires {
        if !program.input_leaf_bindings.contains_key(wire) || values.contains_key(wire) {
            continue;
        }
        let input = program
            .runtime_input_wires
            .iter()
            .position(|candidate| candidate == wire)
            .ok_or("prepared family scalar leaf has no runtime slot")?;
        let slot = program.scalar_slots[wire];
        let words = program.scalar_input_max_words.get(input).copied().unwrap_or(1).max(1);
        let output = allocate_scalar_buffer(backend, region, anchor, device, 1, words)?;
        let mut command = PreparedCommand::new(PreparedOperation::ScalarUpload {
            command: Arc::clone(&output),
            input,
            wire: *wire,
            kind: program.wire_types[wire].clone(),
            device,
        });
        let root =
            program.input_leaf_bindings.get(wire).map(|binding| binding.root).unwrap_or(*wire);
        let topology = program
            .topology
            .nodes
            .iter()
            .find(|node| program.node_bindings[&node.id].1.contains(&root))
            .ok_or("prepared scalar input has no topology node")?;
        command.apply_topology(topology);
        commands.push(command);
        values.insert(*wire, (output.clone(), 0));
        slots.insert(slot, (output, 0));
    }
    for node in &program.topology.nodes {
        let (arguments, outputs) = &program.node_bindings[&node.id];
        let Some(&wire) = outputs.first() else { continue };
        if !program.device_scalar_wires.contains(&wire) || values.contains_key(&wire) {
            continue;
        }
        let kind = &program.wire_types[&wire];
        if matches!(kind, ConcreteWireType::IndexedFamily { .. }) {
            continue;
        }
        let source = &program.node_sources[&node.id];
        let slot = program.scalar_slots[&wire];
        let output = match slots.get(&slot) {
            Some((owner, _)) => Arc::clone(owner),
            None => {
                let words = capacities.get(&slot).copied().unwrap_or(unused_capacity);
                let owner = allocate_scalar_buffer(backend, region, anchor, device, 1, words)?;
                slots.insert(slot, (Arc::clone(&owner), 0));
                owner
            }
        };
        if let Some(input) = program.runtime_input_wires.iter().position(|input| *input == wire) {
            match runtime_inputs.get(input).ok_or("scalar input missing")? {
                PreparedRuntimeValue::Int(_) |
                PreparedRuntimeValue::Bool(_) |
                PreparedRuntimeValue::Real(_) => {}
                _ => return Err("device scalar input has non-scalar binding".into()),
            };
            let mut command = PreparedCommand::new(PreparedOperation::ScalarUpload {
                command: Arc::clone(&output),
                input,
                wire,
                kind: kind.clone(),
                device,
            });
            command.apply_topology(node);
            commands.push(command);
            values.insert(wire, (output, 0));
            continue;
        }
        let variants = if source.variants.is_empty() {
            std::slice::from_ref(&source.kind)
        } else {
            &source.variants
        };
        let copied = program
            .scalar_commands
            .get(&node.id)
            .is_some_and(|command| command.instructions.is_empty()) &&
            !arguments.is_empty();
        let constants = variants
            .iter()
            .map(|kind| if copied { Ok(None) } else { constant(kind, &source.environment) })
            .collect::<Result<Vec<_>, _>>()?;
        let mut prepared = Vec::new();
        let output_words = output.words();
        for (variant, kind) in variants.iter().enumerate() {
            let mut candidates = Vec::new();
            let (opcode, left, right, bit) = if let Some(selection) =
                program.selection_commands.get(&node.id)
            {
                use super::super::gpu_prepared_lowering::PreparedSelection;
                match selection {
                    PreparedSelection::ScalarStatic { slot } => (
                        GpuPreparedScalarOpcode::Copy,
                        slots.get(slot).cloned().ok_or("scalar candidate is not bound")?,
                        None,
                        0,
                    ),
                    PreparedSelection::ScalarDynamic { candidates: selected, selector } |
                    PreparedSelection::ScalarSelect { candidates: selected, selector } => {
                        candidates = selected
                            .iter()
                            .map(|slot| {
                                slots.get(slot).cloned().ok_or("scalar candidate is not bound")
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        (
                            GpuPreparedScalarOpcode::Select,
                            values
                                .get(selector)
                                .cloned()
                                .ok_or("device scalar selector is not bound")?,
                            None,
                            0,
                        )
                    }
                    _ => return Err("matrix selection is not a scalar command".into()),
                }
            } else if let Some(value) = &constants[variant] {
                let words = match value {
                    PreparedRuntimeValue::Int(value) => value.bits().div_ceil(64) as usize + 1,
                    _ => 1,
                };
                let input = allocate_scalar_buffer(backend, region, anchor, device, 1, words)?;
                stage_runtime_scalar(&input, value)?;
                input.wait()?;
                (GpuPreparedScalarOpcode::Copy, (input, 0), None, 0)
            } else {
                let left = values
                    .get(arguments.first().ok_or("device scalar operand missing")?)
                    .cloned()
                    .ok_or("device scalar source is not bound")?;
                let right = arguments
                    .get(1)
                    .map(|wire| values.get(wire).cloned().ok_or("device scalar RHS is not bound"))
                    .transpose()?;
                let (opcode, bit) = if copied {
                    (GpuPreparedScalarOpcode::Copy, 0)
                } else {
                    match kind {
                        NodeKind::IntBinary(operation) => (
                            match operation {
                                IntBinaryOp::Add => GpuPreparedScalarOpcode::Add,
                                IntBinaryOp::Subtract => GpuPreparedScalarOpcode::Subtract,
                                IntBinaryOp::Multiply => GpuPreparedScalarOpcode::Multiply,
                                IntBinaryOp::Divide => GpuPreparedScalarOpcode::Divide,
                                IntBinaryOp::Remainder => GpuPreparedScalarOpcode::Remainder,
                            },
                            0,
                        ),
                        NodeKind::IntCompare(operation) => (
                            match operation {
                                IntCompareOp::Equal => GpuPreparedScalarOpcode::Equal,
                                IntCompareOp::Less => GpuPreparedScalarOpcode::Less,
                                IntCompareOp::LessEqual => GpuPreparedScalarOpcode::LessEqual,
                            },
                            0,
                        ),
                        NodeKind::BitExtract { bit } => (
                            GpuPreparedScalarOpcode::BitExtract,
                            bit.evaluate(&source.environment)
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or("negative bit index")?,
                        ),
                        NodeKind::IntToReal => (GpuPreparedScalarOpcode::IntToReal, 0),
                        NodeKind::BoolToInt => (GpuPreparedScalarOpcode::BoolToInt, 0),
                        NodeKind::RealBinary(operation) => (
                            match operation {
                                RealBinaryOp::Add => GpuPreparedScalarOpcode::RealAdd,
                                RealBinaryOp::Subtract => GpuPreparedScalarOpcode::RealSubtract,
                                RealBinaryOp::Multiply => GpuPreparedScalarOpcode::RealMultiply,
                                RealBinaryOp::Divide => GpuPreparedScalarOpcode::RealDivide,
                            },
                            0,
                        ),
                        NodeKind::RealSqrt => (GpuPreparedScalarOpcode::RealSqrt, 0),
                        _ => return Err(format!("device scalar operation is not bound: {kind:?}")),
                    }
                };
                (opcode, left, right, bit)
            };
            prepared.push((opcode, left, right, bit, candidates));
        }
        for (variant, (opcode, left, right, bit, candidates)) in prepared.into_iter().enumerate() {
            let bytes = GpuPreparedScalarOp::workspace_bytes(
                left.0.words(),
                right.as_ref().map_or(0, |value| value.0.words()),
                output_words,
                candidates.len(),
            );
            let claims = [GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::BatchWorkspace,
                bytes,
                alignment: 8,
            })];
            let plan =
                bind_prepared_claims(backend, region, anchor.params(), device, &claims, || {
                    GpuPreparedScalarOp::bind(
                        opcode,
                        left,
                        right,
                        (Arc::clone(&output), 0),
                        bit,
                        &candidates,
                    )
                })?;
            let mut command = PreparedCommand::new(PreparedOperation::ScalarOp {
                command: plan,
                wire,
                kind: kind.clone(),
                device,
            });
            command.apply_topology(node);
            command.variant = variant;
            commands.push(command);
        }
        values.insert(wire, (output, 0));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Backend, MemoryArtifactStore, RuntimeValue,
        executor::{ExecutionConfig, execute_with_config},
        transcript::SamplingMode,
    };
    use mxx_ir_core::{
        expr::{IntExpr, ParamEnv},
        graph::{
            Graph, GraphOutput, NodeHandle, SubgraphHandle, ValueHandle,
            with_new_construction_scope,
        },
        node::SequentialLoop,
        types::{ConcreteMatrixType, MatrixType, WireType},
        validate::validate,
    };
    use mxx_primitives::{
        poly::dcrt::params::DCRTPolyParams,
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_threshold_scalar_chain_packs_without_host_boundary() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let q: num_bigint::BigInt = params.modulus().as_ref().clone().into();
        let p = &q * 2u32 + 3u32;
        let ring = MatrixType {
            modulus: IntExpr::Const(q.clone()),
            ring_dimension: IntExpr::constant(n),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let matrix_type = WireType::Matrix(ring.clone());
        let node = |kind, args: Vec<ValueHandle>, ty| {
            NodeHandle::new(kind, args, vec![ty]).output(0).unwrap()
        };
        let input = node(
            NodeKind::Input {
                name: "input".into(),
                wire_type: matrix_type.clone(),
                artifact: None,
            },
            vec![],
            matrix_type.clone(),
        );
        let offset = node(
            NodeKind::Input { name: "offset".into(), wire_type: WireType::Int, artifact: None },
            vec![],
            WireType::Int,
        );
        let decoded = node(
            NodeKind::ThresholdDecode {
                plaintext_modulus: IntExpr::Const(p.clone()),
                length: IntExpr::constant(1),
                output_bool: false,
            },
            vec![input],
            WireType::Int,
        );
        let constant =
            |value: i32| node(NodeKind::ConstantInt(value.into()), vec![], WireType::ConstantInt);
        let sum = node(
            NodeKind::IntBinary(IntBinaryOp::Add),
            vec![decoded, offset.clone()],
            WireType::Int,
        );
        let body = with_new_construction_scope(|scope| {
            let carried = node(
                NodeKind::Input {
                    name: "carried".into(),
                    wire_type: WireType::Int,
                    artifact: None,
                },
                vec![],
                WireType::Int,
            );
            let seven = node(NodeKind::ConstantInt(7.into()), vec![], WireType::ConstantInt);
            let next = node(
                NodeKind::IntBinary(IntBinaryOp::Add),
                vec![carried.clone(), seven],
                WireType::Int,
            );
            SubgraphHandle::new("scalar_carried", scope, vec![carried], vec![next]).unwrap()
        });
        let sum = NodeHandle::sequential_loop(
            body,
            vec![sum],
            vec![WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(5),
                index_slot: 0,
                bindings: vec![],
                carried_count: 1,
            },
        )
        .output(0)
        .unwrap();
        let product = node(
            NodeKind::IntBinary(IntBinaryOp::Multiply),
            vec![sum, constant(-3)],
            WireType::Int,
        );
        let selector = node(
            NodeKind::IntCompare(IntCompareOp::Less),
            vec![offset.clone(), constant(0)],
            WireType::Bool,
        );
        let selector = node(NodeKind::BoolToInt, vec![selector], WireType::Int);
        let incremented = node(
            NodeKind::IntBinary(IntBinaryOp::Add),
            vec![product.clone(), constant(1)],
            WireType::Int,
        );
        let product = node(
            NodeKind::Select { count: IntExpr::constant(2) },
            vec![selector.clone(), product, incremented],
            WireType::Int,
        );
        let quotient = node(
            NodeKind::IntBinary(IntBinaryOp::Divide),
            vec![product.clone(), constant(2)],
            WireType::Int,
        );
        let remainder = node(
            NodeKind::IntBinary(IntBinaryOp::Remainder),
            vec![quotient, constant(11)],
            WireType::Int,
        );
        let bit = node(
            NodeKind::BitExtract { bit: IntExpr::constant(0) },
            vec![remainder],
            WireType::Bool,
        );
        let bit = node(NodeKind::BoolToInt, vec![bit], WireType::Int);
        let predicate =
            node(NodeKind::IntCompare(IntCompareOp::Equal), vec![bit, constant(1)], WireType::Bool);
        let integer = node(NodeKind::BoolToInt, vec![predicate], WireType::Int);
        let predicate = node(
            NodeKind::IntCompare(IntCompareOp::Less),
            vec![constant(0), integer],
            WireType::Bool,
        );
        let zero = node(
            NodeKind::IntCompare(IntCompareOp::Equal),
            vec![constant(0), constant(1)],
            WireType::Bool,
        );
        let width = q.bits() as usize;
        let bits = (0..n as usize * width)
            .map(|bit| if bit % width == 0 { predicate.clone() } else { zero.clone() })
            .collect::<Vec<_>>();
        let family = node(
            NodeKind::FamilyPack { count: IntExpr::constant(bits.len()) },
            bits,
            WireType::IndexedFamily {
                element: Box::new(WireType::Bool),
                count: IntExpr::constant(n as usize * width),
            },
        );
        let packed = node(
            NodeKind::PackPolynomialCoefficients {
                matrix_type: ring,
                coefficient_bits: IntExpr::constant(width),
            },
            vec![family],
            matrix_type.clone(),
        );
        let negated = node(NodeKind::MatrixNegate, vec![packed.clone()], matrix_type.clone());
        let packed = node(
            NodeKind::Select { count: IntExpr::constant(2) },
            vec![selector, packed, negated],
            matrix_type,
        );
        let square = node(
            NodeKind::IntBinary(IntBinaryOp::Multiply),
            vec![product.clone(), product],
            WireType::Int,
        );
        let real = node(NodeKind::IntToReal, vec![square], WireType::Real);
        let real = node(NodeKind::RealSqrt, vec![real], WireType::Real);
        let (graph, _) = Graph::freeze(
            "prepared_device_scalars",
            vec![],
            BTreeMap::from([
                ("offset".into(), GraphOutput { value: offset, confidentiality: None }),
                ("result".into(), GraphOutput { value: packed, confidentiality: None }),
                ("real".into(), GraphOutput { value: real, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let graph = validate(&graph, &ParamEnv::default()).unwrap();
        let source =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let mut reference = crate::backend::poly::cpu_backend([cpu]);
        let decoded = reference.threshold_decode(&source, &p, 1).unwrap().remove(0);
        let ty = ConcreteMatrixType { modulus: q, ring_dimension: n as usize, rows: 1, columns: 1 };
        let mut expected = Vec::new();
        for offset in [-7i32, 9, -11] {
            let product = (&decoded + offset + 35i32) * -3i32 + i32::from(offset < 0);
            let bit = ((&product / 2i32) % 11i32 & num_bigint::BigInt::from(1)) !=
                num_bigint::BigInt::from(0);
            let bits =
                (0..n as usize * width).map(|index| index % width == 0 && bit).collect::<Vec<_>>();
            let matrix = reference.pack_polynomial_coefficients(&ty, &bits, width).unwrap();
            let matrix = if offset < 0 { reference.negate(&matrix).unwrap() } else { matrix };
            expected.push((
                offset,
                GpuDCRTPolyMatrix::from_cpu_matrix(&params, &matrix).to_compact_bytes(),
                (&product * &product).to_f64().unwrap().sqrt(),
            ));
        }
        let source = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &source).into();
        let mut inputs = BTreeMap::from([
            ("input".into(), RuntimeValue::matrix(source)),
            ("offset".into(), RuntimeValue::Int((-7).into())),
        ]);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        backend.warm_up_prepared_graph(&graph, &inputs, &ExecutionConfig::default()).unwrap();
        #[cfg(feature = "gpu-instrumentation")]
        {
            let (program, execution) = backend.prepared_execution_for_test();
            let runtime_inputs = program
                .input_names
                .iter()
                .map(|(name, _)| {
                    super::super::super::fleet::prepared_runtime_value(&inputs[name]).unwrap()
                })
                .collect::<Vec<_>>();
            for _ in 0..300 {
                reset_prepared_gpu_work_counters();
                begin_prepared_gpu_work_gate();
                let output = execution.run_with_runtime_inputs(&runtime_inputs).unwrap();
                end_prepared_gpu_work_gate();
                let mut counters = prepared_gpu_work_counters();
                assert!(counters.production_kernels > 0);
                counters.production_kernels = 0;
                assert_eq!(counters, PreparedGpuWorkCounters::default());
                drop(output);
            }
        }
        let mut store = MemoryArtifactStore::default();
        for (offset, expected, expected_real) in expected {
            inputs.insert("offset".into(), RuntimeValue::Int(offset.into()));
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs.clone(),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::Int(actual_offset) =
                result.materialize_output("offset", &mut backend, &mut store).unwrap()
            else {
                panic!("root scalar export")
            };
            assert_eq!(actual_offset, &num_bigint::BigInt::from(offset));
            let RuntimeValue::Matrix(actual) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("packed matrix")
            };
            assert_eq!(backend.matrix_to_bytes(actual).unwrap(), expected);
            let RuntimeValue::Real(actual) =
                result.materialize_output("real", &mut backend, &mut store).unwrap()
            else {
                panic!("real scalar")
            };
            assert_eq!(*actual, expected_real);
        }
        inputs.insert("offset".into(), RuntimeValue::Int(num_bigint::BigInt::from(1) << 128usize));
        let error = execute_with_config(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .err()
        .expect("oversized root integer must not write fixed staging");
        assert!(error.to_string().contains("exceeds warmup width"));
    }
}
