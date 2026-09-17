//! Warmup binding of scalar chains that consume device-produced values.
use super::{
    super::gpu_prepared_lowering::{GpuPreparation, ScalarValue},
    *,
};
use mxx_ir_core::{
    node::{IntBinaryOp, IntCompareOp, RealBinaryOp},
    types::ConcreteWireType,
};
use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedScalarOpcode;

pub(super) fn stage_runtime_scalar(
    buffer: &GpuPreparedScalarBuffer,
    value: &PreparedRuntimeValue,
) -> Result<(), String> {
    buffer.upload(|words, width| {
        words.fill(0);
        match value {
            PreparedRuntimeValue::Int(value) => {
                let required = required_scalar_words(value)?;
                if required > width {
                    return Err("prepared scalar input exceeds its fixed warmup capacity".into());
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

/// Number of little-endian words needed for a signed two's-complement value.
/// The extra sign word is intentional: it keeps both positive values whose
/// high bit is set and their negative counterparts representable without
/// treating the warmup value as a semantic width limit.
pub(super) fn required_scalar_words(value: &num_bigint::BigInt) -> Result<usize, String> {
    value
        .bits()
        .try_into()
        .map_err(|_| "scalar integer width exceeds host capacity".to_owned())
        .and_then(|bits: usize| {
            bits.div_ceil(64).checked_add(1).ok_or_else(|| "scalar capacity overflow".into())
        })
}

pub(super) fn allocate_scalar_buffer_for_wire(
    resources: &super::super::gpu_prepared_lowering::PreparedResolvedResources,
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    matrix_context_owners: &[Arc<GpuDCRTPolyMatrix>],
    wire: WireRef,
    instance: usize,
    device: i32,
) -> Result<Arc<GpuPreparedScalarBuffer>, String> {
    let resolved = resources
        .scalar_buffers
        .iter()
        .find(|buffer| {
            buffer.plan.wire == wire &&
                buffer.plan.instance == instance &&
                resources
                    .finalized_matrices
                    .identity(buffer.plan.owner.matrix_id)
                    .is_some_and(|identity| identity.physical.device == device)
        })
        .ok_or_else(|| format!("prepared scalar buffer {wire:?} has no resolved descriptor"))?;
    let slots = resolved
        .slots
        .iter()
        .cloned()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("prepared scalar buffer {wire:?} has an unresolved slot"))?;
    let owner_identity =
        resources.finalized_matrices.identity(resolved.plan.owner.matrix_id).ok_or_else(|| {
            format!("prepared scalar buffer {wire:?} has no canonical owner identity")
        })?;
    let context_owner = matrix_context_owners
        .iter()
        .find(|candidate| {
            candidate.params().device_ids().contains(&device) &&
                candidate.params().context_identity() == owner_identity.physical.context_identity &&
                candidate.level() == owner_identity.physical.level &&
                (candidate.is_ntt() ==
                    (owner_identity.physical.format ==
                        super::super::gpu_prepared_lowering::PreparedFormat::Evaluation))
        })
        .ok_or_else(|| {
            format!(
                "prepared scalar buffer {wire:?} has no matrix context owner for device {device}"
            )
        })?;
    allocate_scalar_buffer_with_layout(
        region,
        context_owner,
        resolved.plan.count,
        resolved.plan.words,
        resolved.layout.clone(),
        &slots,
    )
}

pub(super) fn allocate_scalar_buffer_with_layout(
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    context_owner: &Arc<GpuDCRTPolyMatrix>,
    count: usize,
    words: usize,
    layout: mxx_primitives::matrix::gpu_dcrt_poly::PreparedPlanLayout,
    slots: &[super::super::gpu_prepared_lowering::PreparedSlotRef],
) -> Result<Arc<GpuPreparedScalarBuffer>, String> {
    super::bind_prepared_slots(region, slots, || {
        GpuPreparedScalarBuffer::bind_with_layout(Arc::clone(context_owner), count, words, layout)
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

/// Evaluate only magnitude bounds at warmup. The fixed replay walks these
/// abstract values across every finite structural variant, so a loop's integer
/// capacity reflects all iterations without guessing a runtime margin.
pub(crate) fn scalar_capacities(
    program: &GpuPreparation,
    runtime_inputs: &[PreparedRuntimeValue],
) -> Result<BTreeMap<usize, usize>, String> {
    use super::super::gpu_prepared_lowering::PreparedReplayStep;
    use num_bigint::BigUint;

    fn visit(
        program: &GpuPreparation,
        runtime_inputs: &[PreparedRuntimeValue],
        steps: &[PreparedReplayStep],
        parent: Option<usize>,
        bounds: &mut BTreeMap<usize, BigUint>,
        capacities: &mut BTreeMap<usize, usize>,
    ) -> Result<(), String> {
        for step in steps {
            match step {
                PreparedReplayStep::Node(id) => {
                    let (arguments, outputs) = &program.node_bindings[id];
                    let source = &program.node_sources[id];
                    let has_scalar_output = outputs
                        .iter()
                        .flat_map(|wire| {
                            std::iter::once(*wire).chain(
                                super::super::gpu_prepared_lowering::family_leaf_wires(
                                    program, *wire,
                                ),
                            )
                        })
                        .any(|wire| program.scalar_slots.contains_key(&wire));
                    if !has_scalar_output {
                        continue;
                    }
                    let kind = parent
                        .and_then(|index| source.variant_indices.get(index))
                        .and_then(|index| source.variants.get(*index))
                        .unwrap_or(&source.kind);
                    // Input-family leaves were seeded from the normalized
                    // runtime contract before replay. They are roots, not a
                    // derived operation, so there is no operation-specific
                    // magnitude to compute here.
                    if matches!(kind, NodeKind::Input { .. }) &&
                        outputs
                            .iter()
                            .flat_map(|wire| {
                                std::iter::once(*wire).chain(
                                    super::super::gpu_prepared_lowering::family_leaf_wires(
                                        program, *wire,
                                    ),
                                )
                            })
                            .filter_map(|wire| program.scalar_slots.get(&wire))
                            .all(|slot| bounds.contains_key(slot))
                    {
                        continue;
                    }
                    let operand = |index: usize| -> Result<BigUint, String> {
                        arguments
                            .get(index)
                            .and_then(|wire| program.scalar_slots.get(wire))
                            .and_then(|slot| bounds.get(slot))
                            .cloned()
                            .ok_or_else(|| {
                                format!(
                                    "device scalar node {id} operand {index} has no finite warmup projection"
                                )
                            })
                    };
                    let magnitude = if program
                        .scalar_commands
                        .get(id)
                        .is_some_and(|command| command.instructions.is_empty()) &&
                        !arguments.is_empty()
                    {
                        operand(0)?
                    } else if let Some(wire) = outputs.first().filter(|wire| {
                        program.inputs.contains(wire) && program.scalar_slots.contains_key(wire)
                    }) {
                        let input = program
                            .runtime_input_wires
                            .iter()
                            .position(|input| input == wire)
                            .ok_or_else(|| format!("device scalar node {id} input has no slot"))?;
                        match runtime_inputs.get(input) {
                            Some(PreparedRuntimeValue::Int(value)) => value.magnitude().clone(),
                            Some(PreparedRuntimeValue::Real(_) | PreparedRuntimeValue::Bool(_)) => {
                                BigUint::from(1u8)
                            }
                            _ => {
                                return Err(format!(
                                    "device scalar node {id} input has no finite projection"
                                ));
                            }
                        }
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
                            NodeKind::ThresholdDecode { output_bool: true, .. } |
                            NodeKind::IntCompare(_) |
                            NodeKind::BitExtract { .. } |
                            NodeKind::BoolToInt |
                            NodeKind::IntToReal |
                            NodeKind::RealBinary(_) |
                            NodeKind::RealSqrt |
                            NodeKind::ConstantBool(_) |
                            NodeKind::ConstantReal(_) => BigUint::from(1u8),
                            NodeKind::ExtractCoefficient {
                                canonical_input_exclusive_upper,
                                ..
                            } => {
                                if let Some(upper) = canonical_input_exclusive_upper.clone() {
                                    upper
                                } else {
                                    arguments
                                        .first()
                                        .and_then(|wire| program.wire_types[wire].matrix_type())
                                        .map(|matrix| matrix.modulus.magnitude().clone())
                                        .ok_or_else(|| {
                                            "coefficient extraction input has no matrix modulus"
                                                .to_owned()
                                        })?
                                }
                            }
                            NodeKind::PolynomialValues { .. } => arguments
                                .first()
                                .and_then(|wire| program.wire_types[wire].matrix_type())
                                .map(|matrix| matrix.modulus.magnitude().clone())
                                .ok_or_else(|| {
                                    "polynomial values input has no matrix modulus".to_owned()
                                })?,
                            NodeKind::ConstantInt(value) => value.magnitude().clone(),
                            NodeKind::EvaluateInt(value) => value
                                .evaluate(&source.environment)
                                .map_err(|error| error.to_string())?
                                .magnitude()
                                .clone(),
                            NodeKind::FamilyPack { .. } => arguments
                                .iter()
                                .enumerate()
                                .map(|(index, _)| operand(index))
                                .collect::<Result<Vec<_>, _>>()?
                                .into_iter()
                                .max()
                                .ok_or_else(|| {
                                    format!(
                                        "device scalar node {id} FamilyPack has no scalar members"
                                    )
                                })?,
                            NodeKind::IntBinary(operation) => {
                                let left = operand(0)?;
                                let right = operand(1)?;
                                match operation {
                                    IntBinaryOp::Add | IntBinaryOp::Subtract => left + right,
                                    IntBinaryOp::Multiply => left * right,
                                    IntBinaryOp::Divide => left,
                                    IntBinaryOp::Remainder => left.min(right),
                                }
                            }
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
                                    .ok_or_else(|| {
                                        format!(
                                            "device scalar node {id} selection has no finite candidate projection"
                                        )
                                    })?
                            }
                            _ => {
                                return Err(format!(
                                    "device scalar node {id} ({kind:?}) has no finite warmup projection"
                                ));
                            }
                        }
                    };
                    for wire in outputs.iter().flat_map(|wire| {
                        std::iter::once(*wire).chain(
                            super::super::gpu_prepared_lowering::family_leaf_wires(program, *wire),
                        )
                    }) {
                        let Some(&slot) = program.scalar_slots.get(&wire) else { continue };
                        let words = match program.wire_types[&wire] {
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
                PreparedReplayStep::Subgraph { body, .. } => {
                    visit(program, runtime_inputs, body, parent, bounds, capacities)?
                }
                PreparedReplayStep::Parallel { counts, waves, .. } => {
                    let active =
                        parent.and_then(|index| counts.get(index)).copied().unwrap_or(usize::MAX);
                    for body in waves.iter().flat_map(|wave| wave.iter()).take(active) {
                        visit(
                            program,
                            runtime_inputs,
                            std::slice::from_ref(body),
                            parent,
                            bounds,
                            capacities,
                        )?;
                    }
                }
                PreparedReplayStep::Sequential {
                    count,
                    counts,
                    offsets,
                    variants,
                    variant_indices,
                    ..
                } => {
                    let count =
                        parent.and_then(|index| counts.get(index)).copied().unwrap_or(*count);
                    let base = parent
                        .and_then(|index| offsets.get(index))
                        .copied()
                        .unwrap_or_else(|| parent.unwrap_or(0).saturating_mul(count));
                    for iteration in 0..count {
                        visit(
                            program,
                            runtime_inputs,
                            variants
                                .get(
                                    *variant_indices
                                        .get(base + iteration)
                                        .ok_or("prepared sequential variant index missing")?,
                                )
                                .ok_or("prepared sequential variant is out of bounds")?,
                            Some(base + iteration),
                            bounds,
                            capacities,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
    let mut capacities = BTreeMap::new();
    let mut bounds = BTreeMap::new();
    seed_runtime_input_projections(program, runtime_inputs, &mut bounds, &mut capacities)?;
    for (&slot, value) in &program.scalar_initializers {
        let magnitude = match value {
            ScalarValue::Int(value) => value.magnitude().clone(),
            ScalarValue::Real(_) | ScalarValue::Bool(_) => BigUint::from(1u8),
            ScalarValue::Runtime(_) | ScalarValue::Slot(_) => {
                return Err(format!("scalar initializer {slot} is not concrete"));
            }
        };
        capacities
            .entry(slot)
            .and_modify(|width| *width = (*width).max(magnitude.bits().div_ceil(64) as usize + 1))
            .or_insert(magnitude.bits().div_ceil(64) as usize + 1);
        bounds.insert(slot, magnitude);
    }
    visit(program, runtime_inputs, &program.replay, None, &mut bounds, &mut capacities)?;
    // Threshold outputs are device-produced scalars even when their replay
    // boundary is represented by a matrix-dependent command. Their exact
    // width is determined by the plaintext modulus during warmup.
    for node in &program.topology.nodes {
        let Some(source) = program.node_sources.get(&node.id) else { continue };
        let NodeKind::ThresholdDecode { plaintext_modulus, output_bool, .. } = source.kind() else {
            continue;
        };
        let magnitude = if *output_bool {
            BigUint::from(1u8)
        } else {
            plaintext_modulus
                .evaluate(&source.environment)
                .map_err(|error| error.to_string())?
                .magnitude()
                .clone()
        };
        let words = magnitude.bits().div_ceil(64) as usize + 1;
        let Some((_, outputs)) = program.node_bindings.get(&node.id) else { continue };
        let wires = outputs
            .iter()
            .flat_map(|wire| {
                std::iter::once(*wire)
                    .chain(super::super::gpu_prepared_lowering::family_leaf_wires(program, *wire))
            })
            .collect::<Vec<_>>();
        for wire in wires {
            if let Some(slot) = program.scalar_slots.get(&wire).copied() {
                capacities
                    .entry(slot)
                    .and_modify(|capacity| *capacity = (*capacity).max(words))
                    .or_insert(words);
            }
        }
    }
    for wire in &program.device_scalar_wires {
        let slot = *program
            .scalar_slots
            .get(wire)
            .ok_or_else(|| format!("device scalar {wire:?} has no fixed slot"))?;
        if !capacities.contains_key(&slot) {
            return Err(format!("device scalar {wire:?} has no operation-aware warmup projection"));
        }
    }
    Ok(capacities)
}

/// Seed every scalar leaf from the normalized runtime-input contract. Family
/// roots are intentionally absent from this table: `runtime_input_wires` and
/// `input_leaf_bindings` are the immutable root/path expansion produced during
/// warmup, so each leaf receives its exact runtime magnitude before replay
/// derives any operation output.
fn seed_runtime_input_projections(
    program: &GpuPreparation,
    runtime_inputs: &[PreparedRuntimeValue],
    bounds: &mut BTreeMap<usize, num_bigint::BigUint>,
    capacities: &mut BTreeMap<usize, usize>,
) -> Result<(), String> {
    for (index, wire) in program.runtime_input_wires.iter().enumerate() {
        let Some(&slot) = program.scalar_slots.get(wire) else { continue };
        let value = runtime_inputs
            .get(index)
            .ok_or_else(|| format!("runtime scalar input {wire:?} is missing"))?;
        let magnitude = match value {
            PreparedRuntimeValue::Int(value) => value.magnitude().clone(),
            PreparedRuntimeValue::Real(_) => num_bigint::BigUint::from(1u8),
            PreparedRuntimeValue::Bool(value) => num_bigint::BigUint::from(u8::from(*value)),
            _ => return Err(format!("runtime scalar input {wire:?} has an invalid value")),
        };
        let words = match program.wire_types.get(wire) {
            Some(ConcreteWireType::Int | ConcreteWireType::ConstantInt) => {
                magnitude.bits().div_ceil(64) as usize + 1
            }
            Some(_) => 1,
            None => return Err(format!("runtime scalar input {wire:?} has no wire type")),
        };
        capacities
            .entry(slot)
            .and_modify(|capacity| *capacity = (*capacity).max(words))
            .or_insert(words);
        bounds.insert(slot, magnitude);
    }
    Ok(())
}

pub(super) fn prepare_scalar_commands(
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    program: &GpuPreparation,
    runtime_inputs: &[PreparedRuntimeValue],
    matrix_context_owners: &[Arc<GpuDCRTPolyMatrix>],
    device: i32,
    values: &mut BTreeMap<WireRef, (Arc<GpuPreparedScalarBuffer>, usize)>,
    commands: &mut Vec<PreparedCommand>,
    resources: &super::super::gpu_prepared_lowering::PreparedResolvedResources,
    instance: usize,
) -> Result<(), String> {
    // Native descriptors use the scalar width fixed by the warmup contract.
    if !program.device_scalar_wires.is_empty() && program.scalar_projections.is_empty() {
        return Err("prepared device scalar projections are missing at execution binding".into());
    }
    let capacities = program.scalar_projections.clone();
    for wire in &program.device_scalar_wires {
        let slot = program.scalar_slots[wire];
        let required = capacities
            .get(&slot)
            .copied()
            .ok_or_else(|| format!("device scalar {wire:?} has no fixed warmup projection"))?;
        let plan = resources
            .scalar_buffers
            .iter()
            .find(|buffer| buffer.plan.wire == *wire && buffer.plan.instance == instance)
            .ok_or_else(|| format!("device scalar {wire:?} has no prepared buffer plan"))?;
        if plan.plan.words < required {
            return Err(format!(
                "device scalar {wire:?} warmup projection exceeds its prepared width"
            ));
        }
    }
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
        let output = allocate_scalar_buffer_for_wire(
            resources,
            region,
            matrix_context_owners,
            *wire,
            instance,
            device,
        )?;
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
                let owner = allocate_scalar_buffer_for_wire(
                    resources,
                    region,
                    matrix_context_owners,
                    wire,
                    instance,
                    device,
                )?;
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
        if let [Some(value)] = constants.as_slice() {
            stage_runtime_scalar(&output, value)?;
            output.wait()?;
            values.insert(wire, (output, 0));
            continue;
        }
        let mut prepared = Vec::new();
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
                stage_runtime_scalar(&output, value)?;
                output.wait()?;
                (GpuPreparedScalarOpcode::Copy, (Arc::clone(&output), 0), None, 0)
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
            let layout = super::resolved_stage_layout(resources, node.id, instance, device)?;
            let slots = super::resolved_stage_slots(resources, node.id, instance, device)?;
            let plan = super::bind_prepared_slots(region, &slots, || {
                GpuPreparedScalarOp::bind_with_layout(
                    opcode,
                    left,
                    right,
                    (Arc::clone(&output), 0),
                    bit,
                    &candidates,
                    layout,
                )
            })?;
            let mut command = PreparedCommand::new(PreparedOperation::ScalarOp {
                command: plan,
                wire,
                kind: kind.clone(),
                device,
            });
            if resources
                .commands
                .iter()
                .find(|resolved| {
                    resolved.command.node == node.id && resolved.command.instance == instance
                })
                .is_some_and(|resolved| resolved.native.is_none())
            {
                command.disable_schedule_owner();
            }
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
        backend::poly_gpu::gpu_prepared_lowering::PreparedInputLeaf,
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
        types::{ConcreteMatrixType, MatrixType, NodeId, Port, WireRef, WireType},
        validate::validate,
    };
    use mxx_primitives::{
        poly::dcrt::params::DCRTPolyParams,
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    fn nested_family_input_projection_uses_normalized_leaf_order() {
        let root = WireRef { node: NodeId(1), port: Port(0) };
        let nested = WireRef { node: NodeId(u64::MAX), port: Port(0) };
        let first = WireRef { node: NodeId(u64::MAX - 1), port: Port(0) };
        let second = WireRef { node: NodeId(u64::MAX - 2), port: Port(1) };
        let mut program = GpuPreparation::default();
        program.inputs = Box::new([root]);
        program.runtime_input_wires = Box::new([first, second]);
        program.runtime_input_roots = Box::new([0, 0]);
        program
            .input_leaf_bindings
            .insert(first, PreparedInputLeaf { root, path: Box::new([0, 0]) });
        program
            .input_leaf_bindings
            .insert(second, PreparedInputLeaf { root, path: Box::new([0, 1]) });
        program.family_wires.insert(root, Box::new([nested]));
        program.family_wires.insert(nested, Box::new([first, second]));
        program.wire_types.insert(first, ConcreteWireType::Int);
        program.wire_types.insert(second, ConcreteWireType::Int);
        program.scalar_slots.insert(first, 3);
        program.scalar_slots.insert(second, 7);
        let runtime_root =
            PreparedRuntimeValue::Family(Arc::from([PreparedRuntimeValue::Family(Arc::from([
                PreparedRuntimeValue::Int(7.into()),
                PreparedRuntimeValue::Int((num_bigint::BigInt::from(1u8)) << 130usize),
            ]))]));
        let runtime_inputs = super::super::expand_prepared_runtime_inputs(
            &program,
            std::slice::from_ref(&runtime_root),
        )
        .unwrap();
        let mut bounds = BTreeMap::new();
        let mut capacities = BTreeMap::new();
        seed_runtime_input_projections(&program, &runtime_inputs, &mut bounds, &mut capacities)
            .unwrap();
        assert_eq!(bounds[&3], 7u8.into());
        assert_eq!(bounds[&7], (num_bigint::BigUint::from(1u8)) << 130usize);
        assert_eq!(capacities[&3], 2);
        assert_eq!(capacities[&7], 4);
        assert_eq!(program.input_leaf_bindings[&first].path.as_ref(), [0, 0]);
        assert_eq!(program.input_leaf_bindings[&second].path.as_ref(), [0, 1]);
        assert_eq!(scalar_capacities(&program, &runtime_inputs).unwrap(), capacities);
    }

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
            let execution = backend.prepared_execution_for_test();
            for _ in 0..300 {
                reset_prepared_gpu_work_counters();
                begin_prepared_gpu_work_gate();
                let mut sampling = SamplingMode::Fresh;
                let output = execution.run_with_runtime_bindings(&inputs, &mut sampling).unwrap();
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
        assert!(
            execute_with_config(
                &graph,
                &mut backend,
                inputs,
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .is_err(),
            "runtime scalar values beyond the warmup projection must be rejected"
        );
    }
}
