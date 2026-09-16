//! Warmup binding of scalar chains that consume device-produced values.
use super::{super::gpu_prepared_lowering::GpuPreparation, *};
use mxx_ir_core::{
    node::{IntBinaryOp, IntCompareOp, RealBinaryOp},
    types::ConcreteWireType,
};
use mxx_primitives::matrix::gpu_dcrt_poly::{
    GpuPreparedScalarOpcode, GpuScalarCapacityAllocator, GpuScalarCapacityLease,
};

/// Adapter from the prepared runtime ledger to the primitive scalar growth
/// hook.  The returned native-independent lease owns the ledger lifecycle and
/// can therefore outlive this short mutable backend borrow.
pub(crate) struct LedgerScalarCapacityAllocator<'a> {
    ledger: &'a mut crate::gpu_memory::GpuMemoryLedger,
}

struct LedgerScalarCapacityLease {
    lease: Option<crate::gpu_memory::GpuAllocationLease>,
}

impl GpuScalarCapacityLease for LedgerScalarCapacityLease {
    fn commit(&mut self) -> Result<(), String> {
        self.lease
            .as_mut()
            .ok_or("scalar capacity lease already retired")?
            .commit_prepared_scalar()
            .map_err(|error| error.to_string())
    }

    fn cancel(&mut self) -> Result<(), String> {
        self.lease
            .as_mut()
            .ok_or("scalar capacity lease already retired")?
            .cancel_prepared_scalar()
            .map_err(|error| error.to_string())?;
        self.lease = None;
        Ok(())
    }

    fn retire(
        &mut self,
        completion: mxx_primitives::poly::dcrt::gpu::GpuReleaseCompletion,
    ) -> Result<(), String> {
        self.lease
            .as_mut()
            .ok_or("scalar capacity lease already retired")?
            .retire_prepared_scalar(completion)
            .map_err(|error| error.to_string())?;
        self.lease = None;
        Ok(())
    }

    fn quarantine(&mut self) -> Result<(), String> {
        let lease = self.lease.as_mut().ok_or("scalar capacity lease already retired")?;
        lease.quarantine_prepared_scalar().map_err(|error| error.to_string())?;
        self.lease = None;
        Ok(())
    }
}

impl<'a> LedgerScalarCapacityAllocator<'a> {
    pub(crate) fn new(ledger: &'a mut crate::gpu_memory::GpuMemoryLedger) -> Self {
        Self { ledger }
    }
}

impl GpuScalarCapacityAllocator for LedgerScalarCapacityAllocator<'_> {
    fn reserve(
        &mut self,
        _params: &mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams,
        device: i32,
        device_bytes: u64,
        pinned_bytes: u64,
    ) -> Result<Box<dyn GpuScalarCapacityLease>, String> {
        let device = self
            .ledger
            .execution_identities()
            .and_then(|identities| identities.iter().position(|(physical, _)| *physical == device))
            .ok_or("scalar capacity device is not an accepted execution owner")?;
        let reservation = self
            .ledger
            .reserve(
                &[crate::gpu_memory::GpuAllocationRequirement {
                    device,
                    bytes: device_bytes,
                    pinned_bytes,
                }],
                &[],
            )
            .map_err(|error| error.to_string())?;
        let id = *reservation
            .allocations
            .first()
            .ok_or("scalar capacity reservation returned no allocation")?;
        let mut leases = match self.ledger.submit(std::slice::from_ref(&id)) {
            Ok(leases) => leases,
            Err(error) => {
                // `reserve` publishes the charge before leases are handed to
                // the caller.  Explicitly cancel here so a dispatcher error
                // cannot strand a generation's budget.
                let _ = self.ledger.cancel(&reservation.allocations);
                return Err(error.to_string());
            }
        };
        let lease = match leases.pop() {
            Some(lease) => lease,
            None => {
                let _ = self.ledger.cancel(&reservation.allocations);
                return Err("scalar capacity reservation returned no lease".into());
            }
        };
        Ok(Box::new(LedgerScalarCapacityLease { lease: Some(lease) }))
    }
}

pub(super) fn required_runtime_scalar_capacity(
    commands: &[super::PreparedCommand],
    inputs: &[PreparedRuntimeValue],
) -> Result<usize, String> {
    let mut required = 1;
    for value in inputs {
        if let PreparedRuntimeValue::Int(value) = value {
            required = required.max(required_scalar_words(value)?);
        }
    }
    for command in commands {
        if let super::PreparedOperation::ScalarOp { command, .. } = &command.operation {
            required = required.max(
                command
                    .inputs()
                    .chain(std::iter::once(command.output()))
                    .map(|buffer| buffer.words())
                    .max()
                    .unwrap_or(1),
            );
        }
    }
    let scalar_ops = commands
        .iter()
        .filter(|command| matches!(command.operation, super::PreparedOperation::ScalarOp { .. }))
        .count();
    if required > 1 {
        required = required
            .checked_mul(scalar_ops.checked_add(1).ok_or("scalar capacity overflow")?)
            .and_then(|words| words.checked_add(scalar_ops))
            .ok_or("scalar capacity overflow")?;
    }
    Ok(required)
}

pub(super) fn ensure_runtime_scalar_capacity<'a, 'b>(
    commands: &[super::PreparedCommand],
    inputs: &[PreparedRuntimeValue],
    mut allocator: Option<&'a mut (dyn GpuScalarCapacityAllocator + 'b)>,
) -> Result<usize, String> {
    let required = required_runtime_scalar_capacity(commands, inputs)?;
    for command in commands {
        match &command.operation {
            super::PreparedOperation::ScalarUpload { command, .. } => {
                if required > command.words() {
                    command.ensure_capacity(
                        required,
                        allocator.as_deref_mut().ok_or("scalar capacity allocator unavailable")?,
                    )?;
                }
            }
            super::PreparedOperation::ScalarOp { command, .. } => {
                for buffer in command.inputs() {
                    if required > buffer.words() {
                        buffer.ensure_capacity(
                            required,
                            allocator
                                .as_deref_mut()
                                .ok_or("scalar capacity allocator unavailable")?,
                        )?;
                    }
                }
                if required > command.output().words() {
                    command.output().ensure_capacity(
                        required,
                        allocator.as_deref_mut().ok_or("scalar capacity allocator unavailable")?,
                    )?;
                }
            }
            super::PreparedOperation::Threshold { command, .. } => {
                if required > command.output().words() {
                    command.output().ensure_capacity(
                        required,
                        allocator.as_deref_mut().ok_or("scalar capacity allocator unavailable")?,
                    )?;
                }
            }
            super::PreparedOperation::ScalarMatrixSelect { .. } |
            super::PreparedOperation::ScalarPack { .. } |
            _ => {}
        }
    }
    Ok(required)
}

pub(super) fn stage_runtime_scalar(
    buffer: &GpuPreparedScalarBuffer,
    value: &PreparedRuntimeValue,
) -> Result<(), String> {
    if let PreparedRuntimeValue::Int(value) = value {
        let _ = required_scalar_words(value)?;
    }
    buffer.upload(|words, _| {
        words.fill(0);
        match value {
            PreparedRuntimeValue::Int(value) => {
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
    anchor: &Arc<GpuDCRTPolyMatrix>,
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
                buffer.plan.owner.device == device
        })
        .ok_or_else(|| format!("prepared scalar buffer {wire:?} has no resolved descriptor"))?;
    let slots = resolved
        .slots
        .iter()
        .copied()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("prepared scalar buffer {wire:?} has an unresolved slot"))?;
    allocate_scalar_buffer_with_layout(
        region,
        anchor,
        resolved.plan.count,
        resolved.plan.words,
        resolved.layout.clone(),
        &slots,
    )
}

pub(super) fn allocate_scalar_buffer_with_layout(
    region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    anchor: &Arc<GpuDCRTPolyMatrix>,
    count: usize,
    words: usize,
    layout: mxx_primitives::matrix::gpu_dcrt_poly::PreparedPlanLayout,
    slots: &[super::super::gpu_prepared_lowering::PreparedSlotRef],
) -> Result<Arc<GpuPreparedScalarBuffer>, String> {
    super::bind_prepared_slots(region, slots, || {
        GpuPreparedScalarBuffer::bind_with_layout(Arc::clone(anchor), count, words, layout)
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
fn scalar_capacities(
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
                        match runtime_inputs.get(input) {
                            Some(PreparedRuntimeValue::Int(value)) => value.magnitude().clone(),
                            _ => BigUint::from(0u8),
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
                            runtime_inputs,
                            &banks[iteration & 1],
                            Some(base + iteration),
                            bounds,
                            capacities,
                        )?;
                    }
                    if count % 2 == 1 {
                        visit(
                            program,
                            runtime_inputs,
                            tail,
                            Some(base + count - 1),
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
    visit(program, runtime_inputs, &program.replay, None, &mut BTreeMap::new(), &mut capacities)?;
    Ok(capacities)
}

pub(super) fn prepare_scalar_commands(
    region: &mut Arc<crate::gpu_memory::GpuMemoryRegion>,
    program: &GpuPreparation,
    runtime_inputs: &[PreparedRuntimeValue],
    anchor: &Arc<GpuDCRTPolyMatrix>,
    device: i32,
    values: &mut BTreeMap<WireRef, (Arc<GpuPreparedScalarBuffer>, usize)>,
    commands: &mut Vec<PreparedCommand>,
    resources: &super::super::gpu_prepared_lowering::PreparedResolvedResources,
    instance: usize,
) -> Result<(), String> {
    // Initial native descriptors use the structural scalar width. Runtime
    // BigInt magnitude is admitted separately by `ensure_runtime_scalar_capacity`
    // immediately before replay, so warmup must not turn its representative
    // input into a permanent width contract.
    let capacities = scalar_capacities(program, runtime_inputs)?;
    let _ = capacities;
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
        let output =
            allocate_scalar_buffer_for_wire(resources, region, anchor, *wire, instance, device)?;
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
                    resources, region, anchor, wire, instance, device,
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
                let input = allocate_scalar_buffer_for_wire(
                    resources, region, anchor, wire, instance, device,
                )?;
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
    use num_bigint::BigInt;

    #[test]
    fn scalar_capacity_includes_signed_sign_word() {
        assert_eq!(required_scalar_words(&BigInt::from(0)).unwrap(), 1);
        assert_eq!(required_scalar_words(&BigInt::from(-1)).unwrap(), 2);
        assert_eq!(required_scalar_words(&(BigInt::from(1u8) << 63)).unwrap(), 2);
        let negative: BigInt = -(BigInt::from(1u8) << 63usize);
        assert_eq!(required_scalar_words(&negative).unwrap(), 2);
    }

    #[test]
    fn scalar_capacity_is_monotonic_for_repeated_widths() {
        let narrow = required_scalar_words(&BigInt::from(7)).unwrap();
        let wide = required_scalar_words(&(BigInt::from(1u8) << 511)).unwrap();
        let wider = required_scalar_words(&(BigInt::from(1u8) << 1023)).unwrap();
        assert!(narrow <= wide && wide <= wider);
        assert_eq!(required_scalar_words(&BigInt::from(7)).unwrap(), narrow);
        let negative: BigInt = -(BigInt::from(1u8) << 1023usize);
        assert_eq!(required_scalar_words(&negative).unwrap(), wider);
    }
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
            let execution = backend.prepared_execution_for_test();
            for _ in 0..300 {
                reset_prepared_gpu_work_counters();
                begin_prepared_gpu_work_gate();
                let mut sampling = SamplingMode::Fresh;
                let output =
                    execution.run_with_runtime_bindings(&inputs, &mut sampling, None).unwrap();
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
        execute_with_config(
            &graph,
            &mut backend,
            inputs,
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .expect("arbitrary-precision root integer grows its execution-local scalar backing");
    }
}
