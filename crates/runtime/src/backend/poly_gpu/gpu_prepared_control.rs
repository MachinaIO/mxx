//! Fixed control commands for prepared execution.
//!
//! Control work is decided during warmup.  Replay receives only already lowered
//! scalar bytecode and fixed locations; it never walks IR, resolves parameters,
//! allocates storage, or builds a schedule.

use super::gpu_prepared_lowering::{
    FixedCopy, GpuPreparation, PreparedOperation, PreparedScalar, ScalarOpcode, ScalarValue,
    ValueLocation,
};
use mxx_ir_core::node::{IntBinaryOp, IntCompareOp, RealBinaryOp};
use num_bigint::BigInt;
use num_traits::ToPrimitive;

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedScalarCommand {
    constants: Box<[ScalarValue]>,
    instructions: Box<[super::gpu_prepared_lowering::ScalarInstruction]>,
    result: usize,
    result_slot: usize,
    value_count: usize,
}

impl PreparedScalarCommand {
    pub fn from_lowered(scalar: PreparedScalar, result_slot: usize) -> Self {
        let value_count = scalar.constants.len() + scalar.instructions.len();
        Self {
            constants: scalar.constants,
            instructions: scalar.instructions,
            result: scalar.result,
            result_slot,
            value_count,
        }
    }

    pub fn value_count(&self) -> usize {
        self.value_count
    }

    /// Execute fixed scalar bytecode into caller-owned storage.  The caller
    /// keeps `scratch` for the whole prepared instance, so this path performs
    /// no vector or schedule allocation.
    pub fn execute<'a>(
        &self,
        runtime: &[ScalarValue],
        slots: &[ScalarValue],
        scratch: &'a mut [ScalarValue],
    ) -> Result<&'a ScalarValue, String> {
        if scratch.len() < self.value_count {
            return Err("prepared scalar scratch is too small".into());
        }
        for (slot, value) in self.constants.iter().enumerate() {
            scratch[slot] = match value {
                ScalarValue::Runtime(index) => runtime
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| "prepared scalar runtime operand is missing".to_owned())?,
                ScalarValue::Slot(index) => slots
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| "prepared scalar slot operand is missing".to_owned())?,
                value => value.clone(),
            };
        }
        for (instruction_index, instruction) in self.instructions.iter().enumerate() {
            let output = self.constants.len() + instruction_index;
            let mut operands = [None; 2];
            if instruction.operands.len() > operands.len() {
                return Err("prepared scalar operand count exceeds fixed command width".into());
            }
            for (index, operand) in instruction.operands.iter().enumerate() {
                operands[index] =
                    Some(scratch.get(*operand).ok_or("prepared scalar operand is missing")?);
            }
            scratch[output] = evaluate(instruction.opcode, &operands)?;
        }
        scratch.get(self.result).ok_or_else(|| "prepared scalar result is missing".to_owned())
    }
}

fn evaluate(
    opcode: ScalarOpcode,
    operands: &[Option<&ScalarValue>; 2],
) -> Result<ScalarValue, String> {
    let operand = |index: usize| {
        operands
            .get(index)
            .and_then(Option::as_ref)
            .copied()
            .ok_or_else(|| "prepared scalar operand count is invalid".to_owned())
    };
    match opcode {
        ScalarOpcode::IntBinary(operation) => {
            let left = int(operand(0)?)?;
            let right = int(operand(1)?)?;
            let value = match operation {
                IntBinaryOp::Add => left + right,
                IntBinaryOp::Subtract => left - right,
                IntBinaryOp::Multiply => left * right,
                IntBinaryOp::Divide => {
                    if right == BigInt::from(0) {
                        return Err("prepared scalar division by zero".into());
                    }
                    left / right
                }
                IntBinaryOp::Remainder => {
                    if right == BigInt::from(0) {
                        return Err("prepared scalar remainder by zero".into());
                    }
                    left % right
                }
            };
            Ok(ScalarValue::Int(value))
        }
        ScalarOpcode::IntCompare(operation) => {
            let left = int(operand(0)?)?;
            let right = int(operand(1)?)?;
            let value = match operation {
                IntCompareOp::Equal => left == right,
                IntCompareOp::Less => left < right,
                IntCompareOp::LessEqual => left <= right,
            };
            Ok(ScalarValue::Bool(value))
        }
        ScalarOpcode::BitExtract => {
            let value = int(operand(0)?)?;
            let bit = int(operand(1)?)?
                .to_usize()
                .ok_or_else(|| "prepared scalar bit index is invalid".to_owned())?;
            Ok(ScalarValue::Bool(((value >> bit) & BigInt::from(1)) == BigInt::from(1)))
        }
        ScalarOpcode::IntToReal => Ok(ScalarValue::Real(
            int(operand(0)?)?
                .to_f64()
                .ok_or_else(|| "prepared scalar integer cannot convert to real".to_owned())?,
        )),
        ScalarOpcode::BoolToInt => Ok(ScalarValue::Int(if bool_value(operand(0)?)? {
            BigInt::from(1)
        } else {
            BigInt::from(0)
        })),
        ScalarOpcode::RealBinary(operation) => {
            let left = real(operand(0)?)?;
            let right = real(operand(1)?)?;
            let value = match operation {
                RealBinaryOp::Add => left + right,
                RealBinaryOp::Subtract => left - right,
                RealBinaryOp::Multiply => left * right,
                RealBinaryOp::Divide => left / right,
            };
            Ok(ScalarValue::Real(value))
        }
        ScalarOpcode::RealSqrt => Ok(ScalarValue::Real(real(operand(0)?)?.sqrt())),
    }
}

fn int(value: &ScalarValue) -> Result<BigInt, String> {
    match value {
        ScalarValue::Int(value) => Ok(value.clone()),
        _ => Err("prepared scalar expected integer".into()),
    }
}

fn real(value: &ScalarValue) -> Result<f64, String> {
    match value {
        ScalarValue::Real(value) => Ok(*value),
        _ => Err("prepared scalar expected real".into()),
    }
}

fn bool_value(value: &ScalarValue) -> Result<bool, String> {
    match value {
        ScalarValue::Bool(value) => Ok(*value),
        _ => Err("prepared scalar expected boolean".into()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedControlCommand {
    Input {
        output: Option<ValueLocation>,
        slot: Option<usize>,
        runtime_index: Option<usize>,
    },
    Constant {
        output: ValueLocation,
        slot: Option<usize>,
        value: Option<ScalarValue>,
    },
    Scalar {
        command: PreparedScalarCommand,
        variants: Box<[PreparedScalarCommand]>,
        variant_indices: Box<[usize]>,
        output: Option<ValueLocation>,
    },
    Alias {
        source: ValueLocation,
        output: ValueLocation,
    },
    FamilyPack {
        members: Box<[ValueLocation]>,
        scalar_members: Box<[usize]>,
        output: Option<ValueLocation>,
    },
    ScalarSelection {
        candidates: Box<[usize]>,
        selector: PreparedSelector,
        output: usize,
    },
    ScalarAlias {
        source: usize,
        output: usize,
    },
    Selection {
        candidates: Box<[ValueLocation]>,
        selector: PreparedSelector,
        output: ValueLocation,
    },
    Noop,
}

/// One warmup-resolved replay step.  The index is into the instance-local
/// control or native command array; replay never reconstructs this ordering
/// from IR nodes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedExecutableCommand {
    Control(usize),
    Native {
        index: usize,
        variant: usize,
        variant_indices: Box<[usize]>,
    },
    Sequential {
        call: mxx_ir_core::types::NodeId,
        count: usize,
        counts: Box<[usize]>,
        offsets: Box<[usize]>,
        banks: [Box<[PreparedExecutableCommand]>; 2],
        tail: Box<[PreparedExecutableCommand]>,
    },
    Subgraph {
        call: Option<mxx_ir_core::types::NodeId>,
        body: Box<[PreparedExecutableCommand]>,
    },
    Parallel {
        call: mxx_ir_core::types::NodeId,
        counts: Box<[usize]>,
        waves: Box<[Box<[PreparedExecutableCommand]>]>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedSelector {
    Runtime(usize),
    Slot(usize),
    Result(usize),
}

fn selector_source(
    program: &GpuPreparation,
    wire: mxx_ir_core::types::WireRef,
    node_id: u32,
) -> Result<PreparedSelector, String> {
    if let Some(slot) = program.scalar_slots.get(&wire) {
        return Ok(PreparedSelector::Slot(*slot));
    }
    let mut result_index = 0;
    for node in &program.topology.nodes {
        if matches!(node.command.operation, PreparedOperation::Scalar) {
            if node.id == wire.node.0 as u32 {
                return Ok(PreparedSelector::Result(result_index));
            }
            result_index += 1;
        }
    }
    Err(format!("selection node {node_id} selector is not a scalar binding"))
}

fn validate_fixed_location_pair(
    source: &ValueLocation,
    output: &ValueLocation,
) -> Result<(), String> {
    if source.device != output.device || source.shape() != output.shape() {
        return Err("prepared fixed locations are incompatible".into());
    }
    Ok(())
}

fn validate_fixed_copies(copies: &[FixedCopy]) -> Result<(), String> {
    if copies
        .iter()
        .any(|copy| validate_fixed_location_pair(&copy.source, &copy.destination).is_err())
    {
        return Err("prepared fixed view copy has incompatible locations".into());
    }
    Ok(())
}

fn validate_family_pack(
    members: &[ValueLocation],
    scalar_members: &[usize],
    output: Option<&ValueLocation>,
) -> Result<(), String> {
    if members.is_empty() && scalar_members.is_empty() {
        return Err("prepared family has no compatible fixed members".into());
    }
    if let Some(output) = output {
        if members.is_empty() || output.device != members[0].device {
            return Err("prepared family has no compatible fixed members".into());
        }
    }
    Ok(())
}

/// Build all non-native fixed commands during warmup.  Native commands remain
/// owned by the native factory; this function only emits control descriptors,
/// so the common factory can concatenate both sets without a second IR pass.
pub fn build_control_commands(
    program: &GpuPreparation,
) -> Result<Box<[PreparedControlCommand]>, String> {
    let mut commands = Vec::new();
    for node in &program.topology.nodes {
        let output = program
            .node_bindings
            .get(&node.id)
            .and_then(|(_, outputs)| outputs.first())
            .and_then(|wire| program.values.get(wire))
            .cloned();
        let command = match &node.command.operation {
            PreparedOperation::Warmup => {
                let output_wire = program
                    .node_bindings
                    .get(&node.id)
                    .and_then(|(_, outputs)| outputs.first())
                    .copied();
                let is_input = output_wire.is_some_and(|wire| program.inputs.contains(&wire));
                let slot = output_wire.and_then(|wire| program.scalar_slots.get(&wire).copied());
                let runtime_index = output_wire.and_then(|wire| {
                    program.runtime_input_wires.iter().position(|input| *input == wire)
                });
                if output_wire.is_some_and(|wire| program.device_scalar_wires.contains(&wire)) {
                    commands.push(PreparedControlCommand::Noop);
                    continue;
                }
                if is_input {
                    PreparedControlCommand::Input { output, slot, runtime_index }
                } else {
                    match output {
                        Some(output) => PreparedControlCommand::Constant {
                            output,
                            slot,
                            value: slot
                                .and_then(|slot| program.scalar_initializers.get(&slot).cloned()),
                        },
                        None => PreparedControlCommand::Noop,
                    }
                }
            }
            PreparedOperation::Scalar => {
                let device_input = program
                    .node_bindings
                    .get(&node.id)
                    .map(|(arguments, _)| {
                        arguments.iter().any(|wire| program.device_scalar_wires.contains(wire))
                    })
                    .unwrap_or(false);
                if device_input {
                    commands.push(PreparedControlCommand::Noop);
                    continue;
                }
                let scalar = program
                    .scalar_commands
                    .get(&node.id)
                    .cloned()
                    .ok_or_else(|| format!("scalar node {} has no warmup bytecode", node.id))?;
                let result_slot = program
                    .node_bindings
                    .get(&node.id)
                    .and_then(|(_, outputs)| outputs.first())
                    .and_then(|wire| program.scalar_slots.get(wire))
                    .copied()
                    .ok_or_else(|| format!("scalar node {} has no result slot", node.id))?;
                PreparedControlCommand::Scalar {
                    command: PreparedScalarCommand::from_lowered(scalar, result_slot),
                    variants: program.node_sources[&node.id]
                        .variants
                        .iter()
                        .map(|kind| {
                            let slots = program.node_bindings[&node.id]
                                .0
                                .iter()
                                .map(|wire| program.scalar_slots[wire])
                                .collect::<Vec<_>>();
                            super::gpu_prepared_lowering::lower_scalar(
                                kind,
                                &program.node_sources[&node.id].environment,
                                &slots,
                            )
                            .map(|scalar| PreparedScalarCommand::from_lowered(scalar, result_slot))
                            .map_err(|error| format!("prepared scalar variant: {error:?}"))
                        })
                        .collect::<Result<Box<[_]>, String>>()?,
                    variant_indices: program.node_sources[&node.id].variant_indices.clone(),
                    output,
                }
            }
            PreparedOperation::Alias => {
                if let Some(view) = program.view_commands.get(&node.id) {
                    match view {
                        super::gpu_prepared_lowering::PreparedView::Alias(location) |
                        super::gpu_prepared_lowering::PreparedView::TransposeAlias(location) => {
                            PreparedControlCommand::Alias {
                                source: location.clone(),
                                output: location.clone(),
                            }
                        }
                        super::gpu_prepared_lowering::PreparedView::FixedCopies(copies) => {
                            validate_fixed_copies(copies)?;
                            return Err(format!(
                                "view node {} requires an owner-bound copy command",
                                node.id
                            ));
                        }
                    }
                } else {
                    let (inputs, outputs) = program
                        .node_bindings
                        .get(&node.id)
                        .ok_or_else(|| format!("alias node {} has no bindings", node.id))?;
                    let source = inputs
                        .first()
                        .and_then(|wire| program.values.get(wire))
                        .cloned()
                        .ok_or_else(|| format!("alias node {} has no source", node.id))?;
                    let output = outputs
                        .first()
                        .and_then(|wire| program.values.get(wire))
                        .cloned()
                        .ok_or_else(|| format!("alias node {} has no output", node.id))?;
                    validate_fixed_location_pair(&source, &output)?;
                    PreparedControlCommand::Alias { source, output }
                }
            }
            PreparedOperation::Selection => {
                if program.selection_commands.get(&node.id).is_some_and(|selection| match selection
                {
                    super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                        selector, ..
                    } |
                    super::gpu_prepared_lowering::PreparedSelection::Select {
                        selector, ..
                    } => program.device_scalar_wires.contains(selector),
                    _ => false,
                }) {
                    commands.push(PreparedControlCommand::Noop);
                    continue;
                }
                let output_wire = program
                    .node_bindings
                    .get(&node.id)
                    .and_then(|(_, outputs)| outputs.first())
                    .copied();
                if output_wire.is_some_and(|wire| program.device_scalar_wires.contains(&wire)) {
                    commands.push(PreparedControlCommand::Noop);
                    continue;
                }
                if let Some(members) =
                    output_wire.and_then(|wire| program.family_members.get(&wire))
                {
                    let scalar_members = program
                        .node_bindings
                        .get(&node.id)
                        .map(|(arguments, _)| {
                            arguments
                                .iter()
                                .filter_map(|wire| program.scalar_slots.get(wire).copied())
                                .collect::<Vec<_>>()
                                .into_boxed_slice()
                        })
                        .unwrap_or_default();
                    validate_family_pack(members, &scalar_members, output.as_ref())?;
                    PreparedControlCommand::FamilyPack {
                        members: members.clone(),
                        scalar_members,
                        output,
                    }
                } else {
                    let selection = program.selection_commands.get(&node.id).ok_or_else(|| {
                        format!("selection node {} has no lowered binding", node.id)
                    })?;
                    match selection {
                        super::gpu_prepared_lowering::PreparedSelection::Static { location } => {
                            let output = output.ok_or_else(|| {
                                format!("selection node {} has no matrix output", node.id)
                            })?;
                            validate_fixed_location_pair(location, &output)?;
                            PreparedControlCommand::Alias { source: location.clone(), output }
                        }
                        super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                            candidates,
                            selector,
                        } |
                        super::gpu_prepared_lowering::PreparedSelection::Select {
                            candidates,
                            selector,
                        } => PreparedControlCommand::Selection {
                            candidates: candidates.clone(),
                            selector: selector_source(program, *selector, node.id)?,
                            output: output.ok_or_else(|| {
                                format!("selection node {} has no matrix output", node.id)
                            })?,
                        },
                        super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { slot } => {
                            PreparedControlCommand::ScalarAlias {
                                source: *slot,
                                output: program
                                    .scalar_slots
                                    .get(&output_wire.ok_or_else(|| {
                                        format!("selection node {} has no output", node.id)
                                    })?)
                                    .copied()
                                    .ok_or_else(|| {
                                        format!("selection node {} has no scalar output", node.id)
                                    })?,
                            }
                        }
                        super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic {
                            candidates,
                            selector,
                        } |
                        super::gpu_prepared_lowering::PreparedSelection::ScalarSelect {
                            candidates,
                            selector,
                        } => PreparedControlCommand::ScalarSelection {
                            candidates: candidates.clone(),
                            selector: selector_source(program, *selector, node.id)?,
                            output: program
                                .scalar_slots
                                .get(&output_wire.ok_or_else(|| {
                                    format!("selection node {} has no output", node.id)
                                })?)
                                .copied()
                                .ok_or_else(|| {
                                    format!("selection node {} has no scalar output", node.id)
                                })?,
                        },
                    }
                }
            }
            PreparedOperation::ParallelLoop | PreparedOperation::SequentialLoop => continue,
            PreparedOperation::Gpu(_) => continue,
        };
        commands.push(command);
    }
    Ok(commands.into_boxed_slice())
}

/// Replay fixed control commands into preallocated per-instance storage.
/// Native commands and this control slice therefore share one command-order
/// boundary without re-reading IR or allocating a temporary value table.
pub fn execute_control_commands(
    commands: &[PreparedControlCommand],
    runtime: &[ScalarValue],
    scratch: &mut [ScalarValue],
    results: &mut [ScalarValue],
    slots: &mut [ScalarValue],
    selections: &mut [usize],
    iteration: Option<usize>,
) -> Result<usize, String> {
    let mut result_count = 0;
    let mut selection_count = 0;
    for command in commands {
        match command {
            PreparedControlCommand::Scalar { command, variants, variant_indices, .. } => {
                let command = iteration
                    .and_then(|index| variant_indices.get(index))
                    .and_then(|index| variants.get(*index))
                    .unwrap_or(command);
                let value = command.execute(runtime, slots, scratch)?;
                let slot = slots
                    .get_mut(command.result_slot)
                    .ok_or_else(|| "prepared scalar result slot is missing".to_owned())?;
                slot.clone_from(value);
                let result = results
                    .get_mut(result_count)
                    .ok_or_else(|| "prepared scalar result storage is too small".to_owned())?;
                result.clone_from(value);
                result_count += 1;
            }
            PreparedControlCommand::Selection { candidates, selector, .. } => {
                let selected = match selector {
                    PreparedSelector::Runtime(index) => match runtime.get(*index) {
                        Some(ScalarValue::Int(value)) => value
                            .to_usize()
                            .ok_or_else(|| "prepared selector is not a usize".to_owned())?,
                        Some(ScalarValue::Bool(value)) => usize::from(*value),
                        _ => return Err("prepared selector is not an integer/bool".into()),
                    },
                    PreparedSelector::Result(index) => match results.get(*index) {
                        Some(ScalarValue::Int(value)) => value
                            .to_usize()
                            .ok_or_else(|| "prepared selector result is not a usize".to_owned())?,
                        Some(ScalarValue::Bool(value)) => usize::from(*value),
                        _ => return Err("prepared selector result is not an integer/bool".into()),
                    },
                    PreparedSelector::Slot(index) => match slots.get(*index) {
                        Some(ScalarValue::Int(value)) => value
                            .to_usize()
                            .ok_or_else(|| "prepared selector slot is not a usize".to_owned())?,
                        Some(ScalarValue::Bool(value)) => usize::from(*value),
                        _ => return Err("prepared selector slot is not an integer/bool".into()),
                    },
                };
                if selected >= candidates.len() {
                    return Err("prepared selection index is outside its fixed candidates".into());
                }
                if let Some(destination) = selections.get_mut(selection_count) {
                    *destination = selected;
                } else {
                    return Err("prepared selection result storage is too small".into());
                }
                selection_count += 1;
            }
            PreparedControlCommand::ScalarSelection { candidates, selector, output } => {
                let selected = match selector {
                    PreparedSelector::Runtime(index) => match runtime.get(*index) {
                        Some(ScalarValue::Int(value)) => value
                            .to_usize()
                            .ok_or_else(|| "prepared scalar selector is not a usize".to_owned())?,
                        Some(ScalarValue::Bool(value)) => usize::from(*value),
                        _ => return Err("prepared scalar selector is not an integer/bool".into()),
                    },
                    PreparedSelector::Result(index) => match results.get(*index) {
                        Some(ScalarValue::Int(value)) => value.to_usize().ok_or_else(|| {
                            "prepared scalar selector result is not a usize".to_owned()
                        })?,
                        Some(ScalarValue::Bool(value)) => usize::from(*value),
                        _ => {
                            return Err(
                                "prepared scalar selector result is not an integer/bool".into()
                            )
                        }
                    },
                    PreparedSelector::Slot(index) => match slots.get(*index) {
                        Some(ScalarValue::Int(value)) => value.to_usize().ok_or_else(|| {
                            "prepared scalar selector slot is not a usize".to_owned()
                        })?,
                        Some(ScalarValue::Bool(value)) => usize::from(*value),
                        _ => {
                            return Err("prepared scalar selector slot is not an integer/bool".into())
                        }
                    },
                };
                let source_slot = candidates.get(selected).ok_or_else(|| {
                    "prepared scalar selection index is outside candidates".to_owned()
                })?;
                let value = slots
                    .get(*source_slot)
                    .cloned()
                    .ok_or_else(|| "prepared scalar selection source is missing".to_owned())?;
                slots
                    .get_mut(*output)
                    .ok_or_else(|| "prepared scalar selection output is missing".to_owned())?
                    .clone_from(&value);
            }
            PreparedControlCommand::ScalarAlias { source, output } => {
                let value = slots
                    .get(*source)
                    .cloned()
                    .ok_or_else(|| "prepared scalar alias source is missing".to_owned())?;
                slots
                    .get_mut(*output)
                    .ok_or_else(|| "prepared scalar alias output is missing".to_owned())?
                    .clone_from(&value);
            }
            PreparedControlCommand::Alias { .. } | PreparedControlCommand::FamilyPack { .. } => {}
            PreparedControlCommand::Input {
                slot: Some(slot),
                runtime_index: Some(runtime_index),
                ..
            } => {
                let value = runtime
                    .get(*runtime_index)
                    .cloned()
                    .ok_or_else(|| "prepared scalar input slot is missing".to_owned())?;
                slots
                    .get_mut(*slot)
                    .ok_or_else(|| "prepared scalar input destination is missing".to_owned())?
                    .clone_from(&value);
            }
            PreparedControlCommand::Input { slot: Some(_), runtime_index: None, .. } => {
                return Err("prepared scalar input has no runtime binding".into());
            }
            PreparedControlCommand::Input { slot: None, .. } => {}
            PreparedControlCommand::Constant { slot: Some(slot), value: Some(value), .. } => {
                slots
                    .get_mut(*slot)
                    .ok_or_else(|| "prepared scalar constant slot is missing".to_owned())?
                    .clone_from(value);
            }
            PreparedControlCommand::Constant { .. } | PreparedControlCommand::Noop => {}
        }
    }
    Ok(result_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::node::{IntBinaryOp, IntCompareOp, RealBinaryOp};

    fn command(
        opcode: ScalarOpcode,
        constants: &[ScalarValue],
        result: usize,
    ) -> PreparedScalarCommand {
        PreparedScalarCommand {
            constants: constants.to_vec().into_boxed_slice(),
            instructions: vec![super::super::gpu_prepared_lowering::ScalarInstruction {
                opcode,
                operands: vec![0, 1].into_boxed_slice(),
            }]
            .into_boxed_slice(),
            result,
            result_slot: 0,
            value_count: constants.len() + 1,
        }
    }

    #[test]
    fn scalar_commands_use_preallocated_scratch() {
        let add = command(
            ScalarOpcode::IntBinary(IntBinaryOp::Add),
            &[ScalarValue::Runtime(0), ScalarValue::Runtime(1)],
            2,
        );
        let mut scratch = vec![ScalarValue::Int(BigInt::from(0)); add.value_count()];
        let slots = vec![ScalarValue::Bool(false); 2];
        let result = add
            .execute(
                &[ScalarValue::Int(BigInt::from(7)), ScalarValue::Int(BigInt::from(5))],
                &slots,
                &mut scratch,
            )
            .unwrap();
        assert_eq!(result, &ScalarValue::Int(BigInt::from(12)));
    }

    #[test]
    fn scalar_commands_preserve_a_1024_bit_product_exactly() {
        let multiply = command(
            ScalarOpcode::IntBinary(IntBinaryOp::Multiply),
            &[ScalarValue::Runtime(0), ScalarValue::Runtime(1)],
            2,
        );
        let value = (BigInt::from(1u8) << 1023usize) + BigInt::from(17u8);
        let expected = &value * &value;
        let mut scratch = vec![ScalarValue::Int(BigInt::from(0)); multiply.value_count()];
        let result = multiply
            .execute(&[ScalarValue::Int(value.clone()), ScalarValue::Int(value)], &[], &mut scratch)
            .unwrap();
        assert_eq!(result, &ScalarValue::Int(expected));
    }

    #[test]
    fn scalar_commands_cover_comparison_and_real_math() {
        let compare = command(
            ScalarOpcode::IntCompare(IntCompareOp::LessEqual),
            &[ScalarValue::Runtime(0), ScalarValue::Runtime(1)],
            2,
        );
        let mut scratch = vec![ScalarValue::Bool(false); compare.value_count()];
        let slots = vec![ScalarValue::Bool(false); 2];
        assert_eq!(
            compare
                .execute(
                    &[ScalarValue::Int(BigInt::from(2)), ScalarValue::Int(BigInt::from(3))],
                    &slots,
                    &mut scratch,
                )
                .unwrap(),
            &ScalarValue::Bool(true)
        );
        let real = command(
            ScalarOpcode::RealBinary(RealBinaryOp::Multiply),
            &[ScalarValue::Runtime(0), ScalarValue::Runtime(1)],
            2,
        );
        let mut scratch = vec![ScalarValue::Real(0.0); real.value_count()];
        let slots = vec![ScalarValue::Bool(false); 2];
        assert_eq!(
            real.execute(&[ScalarValue::Real(2.5), ScalarValue::Real(4.0)], &slots, &mut scratch)
                .unwrap(),
            &ScalarValue::Real(10.0)
        );
    }

    #[test]
    fn scalar_family_selection_copies_the_selected_slot() {
        let command = PreparedControlCommand::ScalarSelection {
            candidates: vec![1usize, 2].into_boxed_slice(),
            selector: PreparedSelector::Slot(0),
            output: 3,
        };
        let mut slots = vec![
            ScalarValue::Int(BigInt::from(1)),
            ScalarValue::Int(BigInt::from(11)),
            ScalarValue::Int(BigInt::from(22)),
            ScalarValue::Int(BigInt::from(0)),
        ];
        let mut scratch: Vec<ScalarValue> = Vec::new();
        let mut results: Vec<ScalarValue> = Vec::new();
        let mut selections: Vec<usize> = Vec::new();
        execute_control_commands(
            std::slice::from_ref(&command),
            &[],
            &mut scratch,
            &mut results,
            &mut slots,
            &mut selections,
            None,
        )
        .unwrap();
        assert_eq!(slots[3], ScalarValue::Int(BigInt::from(22)));
    }

    #[test]
    fn scalar_family_static_selection_aliases_without_reading_selector() {
        let command = PreparedControlCommand::ScalarAlias { source: 1, output: 2 };
        let mut slots =
            vec![ScalarValue::Bool(false), ScalarValue::Real(3.5), ScalarValue::Real(0.0)];
        execute_control_commands(
            std::slice::from_ref(&command),
            &[],
            &mut [],
            &mut [],
            &mut slots,
            &mut [],
            None,
        )
        .unwrap();
        assert_eq!(slots[2], ScalarValue::Real(3.5));
    }

    #[test]
    fn control_replay_dispatches_fixed_scalar_commands() {
        let scalar = command(
            ScalarOpcode::IntBinary(IntBinaryOp::Add),
            &[ScalarValue::Runtime(0), ScalarValue::Runtime(1)],
            2,
        );
        let commands = vec![PreparedControlCommand::Scalar {
            command: scalar,
            variants: Box::new([]),
            variant_indices: Box::new([]),
            output: None,
        }];
        let mut scratch = vec![ScalarValue::Int(BigInt::from(0)); 3];
        let mut results = vec![ScalarValue::Int(BigInt::from(0))];
        let mut slots = vec![ScalarValue::Bool(false); 3];
        let mut selections = Vec::new();
        assert_eq!(
            execute_control_commands(
                &commands,
                &[ScalarValue::Int(BigInt::from(4)), ScalarValue::Int(BigInt::from(9))],
                &mut scratch,
                &mut results,
                &mut slots,
                &mut selections,
                None,
            )
            .unwrap(),
            1
        );
        assert_eq!(results[0], ScalarValue::Int(BigInt::from(13)));
    }

    #[test]
    fn test_gpu_prepared_scalar_variants_reuse_fixed_result_slot() {
        let variant = |value| {
            let mut command = command(
                ScalarOpcode::IntBinary(IntBinaryOp::Add),
                &[ScalarValue::Runtime(0), ScalarValue::Int(BigInt::from(value))],
                2,
            );
            command.result_slot = 1;
            command
        };
        let command = PreparedControlCommand::Scalar {
            command: variant(1),
            variants: vec![variant(1), variant(3)].into_boxed_slice(),
            variant_indices: vec![0, 1, 0].into_boxed_slice(),
            output: None,
        };
        let mut scratch = vec![ScalarValue::Bool(false); 3];
        let mut results = vec![ScalarValue::Bool(false)];
        let mut slots = vec![ScalarValue::Bool(false); 2];
        for iteration in 0..300 {
            execute_control_commands(
                std::slice::from_ref(&command),
                &[ScalarValue::Int(BigInt::from(10))],
                &mut scratch,
                &mut results,
                &mut slots,
                &mut [],
                Some(iteration % 3),
            )
            .unwrap();
            let expected = if iteration % 3 == 1 { 13 } else { 11 };
            assert_eq!(slots[1], ScalarValue::Int(BigInt::from(expected)));
            assert_eq!(results[0], slots[1]);
        }
    }

    #[test]
    fn control_commands_are_fixed_and_non_native() {
        let alias = PreparedControlCommand::Alias {
            source: ValueLocation {
                owner: 1,
                rows: 0..1,
                columns: 0..1,
                level: 0,
                format: super::super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                device: 0,
            },
            output: ValueLocation {
                owner: 1,
                rows: 0..1,
                columns: 0..1,
                level: 0,
                format: super::super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                device: 0,
            },
        };
        assert!(matches!(alias, PreparedControlCommand::Alias { .. }));
    }

    #[test]
    fn warmup_rejects_incompatible_fixed_locations() {
        let location = |device, columns| ValueLocation {
            owner: 1,
            rows: 0..1,
            columns: 0..columns,
            level: 0,
            format: super::super::gpu_prepared_lowering::PreparedFormat::Evaluation,
            device,
        };
        assert!(validate_fixed_location_pair(&location(0, 1), &location(1, 1)).is_err());
        assert!(validate_fixed_location_pair(&location(0, 1), &location(0, 2)).is_err());
        assert!(
            validate_fixed_copies(&[FixedCopy {
                source: location(0, 1),
                destination: location(0, 2)
            },])
            .is_err()
        );
    }

    #[test]
    fn replay_does_not_revalidate_fixed_locations() {
        let location = |device| ValueLocation {
            owner: 1,
            rows: 0..1,
            columns: 0..1,
            level: 0,
            format: super::super::gpu_prepared_lowering::PreparedFormat::Evaluation,
            device,
        };
        let commands = [
            PreparedControlCommand::Alias { source: location(0), output: location(1) },
            PreparedControlCommand::FamilyPack {
                members: Box::new([]),
                scalar_members: vec![0].into_boxed_slice(),
                output: None,
            },
        ];
        let mut slots = vec![ScalarValue::Bool(false)];
        execute_control_commands(&commands, &[], &mut [], &mut [], &mut slots, &mut [], None)
            .unwrap();
    }

    #[test]
    fn selection_replay_uses_prebound_runtime_or_scalar_result() {
        let location = |owner| ValueLocation {
            owner,
            rows: 0..1,
            columns: 0..1,
            level: 0,
            format: super::super::gpu_prepared_lowering::PreparedFormat::Evaluation,
            device: 0,
        };
        let command = PreparedControlCommand::Selection {
            candidates: vec![location(1), location(2)].into_boxed_slice(),
            selector: PreparedSelector::Runtime(0),
            output: location(3),
        };
        let mut scratch = Vec::new();
        let mut results = vec![ScalarValue::Bool(false)];
        let mut slots = vec![ScalarValue::Bool(false); 1];
        let mut selections = vec![0usize];
        execute_control_commands(
            std::slice::from_ref(&command),
            &[ScalarValue::Int(BigInt::from(1))],
            &mut scratch,
            &mut results,
            &mut slots,
            &mut selections,
            None,
        )
        .unwrap();

        let command = PreparedControlCommand::Selection {
            candidates: vec![location(1), location(2)].into_boxed_slice(),
            selector: PreparedSelector::Result(0),
            output: location(3),
        };
        results[0] = ScalarValue::Bool(true);
        execute_control_commands(
            std::slice::from_ref(&command),
            &[],
            &mut scratch,
            &mut results,
            &mut slots,
            &mut selections,
            None,
        )
        .unwrap();
    }
}
