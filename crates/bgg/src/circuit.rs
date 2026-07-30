use crate::{
    BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire, GraphBuilder,
};
use mxx_gadgets::{
    Poly,
    circuit::{GateParamSource, PolyCircuit, PolyGateType, SubCircuitParamValue, gate::GateId},
};
use mxx_graph_ir::node::MatrixBinaryOp;
use num_bigint::BigInt;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CircuitCompileError {
    #[error("gate {gate} references unavailable input gate {input}")]
    MissingInput { gate: usize, input: usize },
    #[error("gate {gate} has an invalid input arity")]
    InvalidArity { gate: usize },
    #[error("gate {gate} requires lowering context for {kind}")]
    MissingGateContext { gate: usize, kind: &'static str },
    #[error("the circuit compiler received more input bundles than the circuit consumes")]
    ExtraInputs,
    #[error("gate {gate}: {source}")]
    Encoding {
        gate: usize,
        #[source]
        source: crate::encoding::EncodingCompileError,
    },
    #[error(transparent)]
    Subgraph(#[from] crate::SubgraphBuildError),
}

#[derive(Clone, Debug)]
pub struct PolyCircuitCompiler {
    pub public_key: BggPublicKeyCompiler,
}

/// Scheme-specific lowering for gates whose concrete construction depends on
/// lookup or slot-transfer preprocessing that is intentionally not stored in a
/// [`PolyCircuit`].
pub trait AdvancedGateLowering<P: Poly, W> {
    fn slot_transfer(
        &mut self,
        builder: &mut GraphBuilder,
        input: &W,
        source_slots: &[(u32, Option<u32>)],
        gate: GateId,
    ) -> Result<W, CircuitCompileError>;

    fn slot_reduce(
        &mut self,
        builder: &mut GraphBuilder,
        inputs: &[W],
        slot_count: usize,
        gate: GateId,
    ) -> Result<W, CircuitCompileError>;

    fn public_lookup(
        &mut self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        lookup_id: usize,
        input: &W,
        gate: GateId,
    ) -> Result<W, CircuitCompileError>;
}

impl PolyCircuitCompiler {
    pub fn compile_public_keys<P: Poly>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggPublicKeyWire,
        inputs: impl IntoIterator<Item = BggPublicKeyWire>,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError> {
        let mut values = BTreeMap::new();
        let mut supplied = inputs.into_iter();
        for (_, gate) in circuit.gates_in_id_order() {
            let gate_id = gate.gate_id.index();
            let value = match &gate.gate_type {
                PolyGateType::Input if gate_id == 0 => one.clone(),
                PolyGateType::Input => supplied
                    .next()
                    .ok_or(CircuitCompileError::MissingInput { gate: gate_id, input: gate_id })?,
                PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                    let [lhs, rhs] = lookup_binary(&values, gate)?;
                    match gate.gate_type {
                        PolyGateType::Add => compile_public_key_binary_template(
                            builder,
                            &self.public_key,
                            MatrixBinaryOp::Add,
                            lhs,
                            rhs,
                        )?,
                        PolyGateType::Sub => compile_public_key_binary_template(
                            builder,
                            &self.public_key,
                            MatrixBinaryOp::Subtract,
                            lhs,
                            rhs,
                        )?,
                        PolyGateType::Mul => compile_public_key_binary_template(
                            builder,
                            &self.public_key,
                            MatrixBinaryOp::Multiply,
                            lhs,
                            rhs,
                        )?,
                        _ => unreachable!(),
                    }
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar = scalar_u32(builder, gate_id, scalar, &input.matrix.matrix_type)?;
                    compile_public_key_scalar_template(
                        builder,
                        &self.public_key,
                        input,
                        &scalar,
                        false,
                    )?
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar =
                        scalar_biguint(builder, gate_id, scalar, &input.matrix.matrix_type)?;
                    compile_public_key_scalar_template(
                        builder,
                        &self.public_key,
                        input,
                        &scalar,
                        true,
                    )?
                }
                other => {
                    return Err(CircuitCompileError::MissingGateContext {
                        gate: gate_id,
                        kind: gate_kind_name(other),
                    });
                }
            };
            values.insert(gate_id, value);
        }
        if supplied.next().is_some() {
            return Err(CircuitCompileError::ExtraInputs);
        }
        collect_outputs(circuit, &values)
    }

    pub fn compile_encodings<P: Poly>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggEncodingWire,
        inputs: impl IntoIterator<Item = BggEncodingWire>,
    ) -> Result<Vec<BggEncodingWire>, CircuitCompileError> {
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        let mut values = BTreeMap::new();
        let mut supplied = inputs.into_iter();
        for (_, gate) in circuit.gates_in_id_order() {
            let gate_id = gate.gate_id.index();
            let value = match &gate.gate_type {
                PolyGateType::Input if gate_id == 0 => one.clone(),
                PolyGateType::Input => supplied
                    .next()
                    .ok_or(CircuitCompileError::MissingInput { gate: gate_id, input: gate_id })?,
                PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                    let [lhs, rhs] = lookup_binary(&values, gate)?;
                    match gate.gate_type {
                        PolyGateType::Add => compile_encoding_binary_template(
                            builder,
                            &compiler,
                            MatrixBinaryOp::Add,
                            lhs,
                            rhs,
                            gate_id,
                        ),
                        PolyGateType::Sub => compile_encoding_binary_template(
                            builder,
                            &compiler,
                            MatrixBinaryOp::Subtract,
                            lhs,
                            rhs,
                            gate_id,
                        ),
                        PolyGateType::Mul => compile_encoding_binary_template(
                            builder,
                            &compiler,
                            MatrixBinaryOp::Multiply,
                            lhs,
                            rhs,
                            gate_id,
                        ),
                        _ => unreachable!(),
                    }?
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar =
                        scalar_u32(builder, gate_id, scalar, &input.pubkey.matrix.matrix_type)?;
                    compile_encoding_scalar_template(builder, &compiler, input, &scalar, false)?
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar =
                        scalar_biguint(builder, gate_id, scalar, &input.pubkey.matrix.matrix_type)?;
                    compile_encoding_scalar_template(builder, &compiler, input, &scalar, true)?
                }
                other => {
                    return Err(CircuitCompileError::MissingGateContext {
                        gate: gate_id,
                        kind: gate_kind_name(other),
                    });
                }
            };
            values.insert(gate_id, value);
        }
        if supplied.next().is_some() {
            return Err(CircuitCompileError::ExtraInputs);
        }
        collect_outputs(circuit, &values)
    }

    pub fn compile_public_keys_with_lowering<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggPublicKeyWire,
        inputs: impl IntoIterator<Item = BggPublicKeyWire>,
        lowering: &mut L,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, BggPublicKeyWire>,
    {
        self.compile_public_keys_scoped(
            builder,
            circuit,
            one,
            inputs.into_iter().collect(),
            &[],
            lowering,
        )
    }

    fn compile_public_keys_scoped<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggPublicKeyWire,
        inputs: Vec<BggPublicKeyWire>,
        bindings: &[SubCircuitParamValue],
        lowering: &mut L,
    ) -> Result<Vec<BggPublicKeyWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, BggPublicKeyWire>,
    {
        let mut values = BTreeMap::new();
        let mut supplied = inputs.into_iter();
        for (_, gate) in circuit.gates_in_id_order() {
            let gate_id = gate.gate_id.index();
            if values.contains_key(&gate_id) {
                continue;
            }
            let value = match &gate.gate_type {
                PolyGateType::Input if gate_id == 0 => one.clone(),
                PolyGateType::Input => supplied
                    .next()
                    .ok_or(CircuitCompileError::MissingInput { gate: gate_id, input: gate_id })?,
                PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                    let [lhs, rhs] = lookup_binary(&values, gate)?;
                    match gate.gate_type {
                        PolyGateType::Add => compile_public_key_binary_template(
                            builder,
                            &self.public_key,
                            MatrixBinaryOp::Add,
                            lhs,
                            rhs,
                        )?,
                        PolyGateType::Sub => compile_public_key_binary_template(
                            builder,
                            &self.public_key,
                            MatrixBinaryOp::Subtract,
                            lhs,
                            rhs,
                        )?,
                        PolyGateType::Mul => compile_public_key_binary_template(
                            builder,
                            &self.public_key,
                            MatrixBinaryOp::Multiply,
                            lhs,
                            rhs,
                        )?,
                        _ => unreachable!(),
                    }
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar =
                        scalar_u32_bound(builder, scalar, bindings, &input.matrix.matrix_type);
                    compile_public_key_scalar_template(
                        builder,
                        &self.public_key,
                        input,
                        &scalar,
                        false,
                    )?
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar =
                        scalar_biguint_bound(builder, scalar, bindings, &input.matrix.matrix_type);
                    compile_public_key_scalar_template(
                        builder,
                        &self.public_key,
                        input,
                        &scalar,
                        true,
                    )?
                }
                PolyGateType::SlotTransfer { src_slots } => {
                    let input = lookup_unary(&values, gate)?;
                    let source_slots = src_slots.resolve_slot_transfer(bindings);
                    lowering.slot_transfer(builder, input, source_slots.as_ref(), gate.gate_id)?
                }
                PolyGateType::SlotReduce { num_slots, .. } => {
                    let inputs = lookup_many(&values, gate)?;
                    lowering.slot_reduce(builder, &inputs, *num_slots, gate.gate_id)?
                }
                PolyGateType::PubLut { lut_id } => {
                    let input = lookup_unary(&values, gate)?;
                    lowering.public_lookup(
                        builder,
                        circuit,
                        lut_id.resolve_public_lookup(bindings),
                        input,
                        gate.gate_id,
                    )?
                }
                PolyGateType::SubCircuitOutput { call_id, .. } => {
                    let info = circuit.sub_circuit_call_info(*call_id);
                    let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                    let child_inputs = flatten_inputs(&values, &info.inputs, gate_id)?;
                    let outputs = self.compile_public_keys_scoped(
                        builder,
                        child.as_ref(),
                        one.clone(),
                        child_inputs,
                        info.param_bindings.as_ref(),
                        lowering,
                    )?;
                    insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                    continue;
                }
                PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                    let info = circuit.summed_sub_circuit_call_info(*summed_call_id);
                    let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                    let mut accumulated: Option<Vec<BggPublicKeyWire>> = None;
                    for (call_inputs, call_bindings) in
                        info.call_inputs.iter().zip(info.param_bindings.iter())
                    {
                        let child_inputs = flatten_inputs(&values, call_inputs, gate_id)?;
                        let outputs = self.compile_public_keys_scoped(
                            builder,
                            child.as_ref(),
                            one.clone(),
                            child_inputs,
                            call_bindings.as_ref(),
                            lowering,
                        )?;
                        if let Some(current) = accumulated.as_mut() {
                            for (sum, output) in current.iter_mut().zip(outputs) {
                                *sum = compile_public_key_binary_template(
                                    builder,
                                    &self.public_key,
                                    MatrixBinaryOp::Add,
                                    sum,
                                    &output,
                                )?;
                            }
                        } else {
                            accumulated = Some(outputs);
                        }
                    }
                    let outputs =
                        accumulated.ok_or(CircuitCompileError::InvalidArity { gate: gate_id })?;
                    insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                    continue;
                }
            };
            values.insert(gate_id, value);
        }
        if supplied.next().is_some() {
            return Err(CircuitCompileError::ExtraInputs);
        }
        collect_outputs(circuit, &values)
    }

    pub fn compile_encodings_with_lowering<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggEncodingWire,
        inputs: impl IntoIterator<Item = BggEncodingWire>,
        lowering: &mut L,
    ) -> Result<Vec<BggEncodingWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, BggEncodingWire>,
    {
        self.compile_encodings_scoped(
            builder,
            circuit,
            one,
            inputs.into_iter().collect(),
            &[],
            lowering,
        )
    }

    fn compile_encodings_scoped<P: Poly, L>(
        &self,
        builder: &mut GraphBuilder,
        circuit: &PolyCircuit<P>,
        one: BggEncodingWire,
        inputs: Vec<BggEncodingWire>,
        bindings: &[SubCircuitParamValue],
        lowering: &mut L,
    ) -> Result<Vec<BggEncodingWire>, CircuitCompileError>
    where
        L: AdvancedGateLowering<P, BggEncodingWire>,
    {
        let compiler = BggEncodingCompiler { public_key: self.public_key.clone() };
        let mut values = BTreeMap::new();
        let mut supplied = inputs.into_iter();
        for (_, gate) in circuit.gates_in_id_order() {
            let gate_id = gate.gate_id.index();
            if values.contains_key(&gate_id) {
                continue;
            }
            let value = match &gate.gate_type {
                PolyGateType::Input if gate_id == 0 => one.clone(),
                PolyGateType::Input => supplied
                    .next()
                    .ok_or(CircuitCompileError::MissingInput { gate: gate_id, input: gate_id })?,
                PolyGateType::Add | PolyGateType::Sub | PolyGateType::Mul => {
                    let [lhs, rhs] = lookup_binary(&values, gate)?;
                    match gate.gate_type {
                        PolyGateType::Add => compile_encoding_binary_template(
                            builder,
                            &compiler,
                            MatrixBinaryOp::Add,
                            lhs,
                            rhs,
                            gate_id,
                        ),
                        PolyGateType::Sub => compile_encoding_binary_template(
                            builder,
                            &compiler,
                            MatrixBinaryOp::Subtract,
                            lhs,
                            rhs,
                            gate_id,
                        ),
                        PolyGateType::Mul => compile_encoding_binary_template(
                            builder,
                            &compiler,
                            MatrixBinaryOp::Multiply,
                            lhs,
                            rhs,
                            gate_id,
                        ),
                        _ => unreachable!(),
                    }?
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar = scalar_u32_bound(
                        builder,
                        scalar,
                        bindings,
                        &input.pubkey.matrix.matrix_type,
                    );
                    compile_encoding_scalar_template(builder, &compiler, input, &scalar, false)?
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let input = lookup_unary(&values, gate)?;
                    let scalar = scalar_biguint_bound(
                        builder,
                        scalar,
                        bindings,
                        &input.pubkey.matrix.matrix_type,
                    );
                    compile_encoding_scalar_template(builder, &compiler, input, &scalar, true)?
                }
                PolyGateType::SlotTransfer { src_slots } => {
                    let input = lookup_unary(&values, gate)?;
                    let source_slots = src_slots.resolve_slot_transfer(bindings);
                    lowering.slot_transfer(builder, input, source_slots.as_ref(), gate.gate_id)?
                }
                PolyGateType::SlotReduce { num_slots, .. } => {
                    let inputs = lookup_many(&values, gate)?;
                    lowering.slot_reduce(builder, &inputs, *num_slots, gate.gate_id)?
                }
                PolyGateType::PubLut { lut_id } => {
                    let input = lookup_unary(&values, gate)?;
                    lowering.public_lookup(
                        builder,
                        circuit,
                        lut_id.resolve_public_lookup(bindings),
                        input,
                        gate.gate_id,
                    )?
                }
                PolyGateType::SubCircuitOutput { call_id, .. } => {
                    let info = circuit.sub_circuit_call_info(*call_id);
                    let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                    let child_inputs = flatten_inputs(&values, &info.inputs, gate_id)?;
                    let outputs = self.compile_encodings_scoped(
                        builder,
                        child.as_ref(),
                        one.clone(),
                        child_inputs,
                        info.param_bindings.as_ref(),
                        lowering,
                    )?;
                    insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                    continue;
                }
                PolyGateType::SummedSubCircuitOutput { summed_call_id, .. } => {
                    let info = circuit.summed_sub_circuit_call_info(*summed_call_id);
                    let child = circuit.registered_sub_circuit_ref(info.sub_circuit_id);
                    let mut accumulated: Option<Vec<BggEncodingWire>> = None;
                    for (call_inputs, call_bindings) in
                        info.call_inputs.iter().zip(info.param_bindings.iter())
                    {
                        let child_inputs = flatten_inputs(&values, call_inputs, gate_id)?;
                        let outputs = self.compile_encodings_scoped(
                            builder,
                            child.as_ref(),
                            one.clone(),
                            child_inputs,
                            call_bindings.as_ref(),
                            lowering,
                        )?;
                        if let Some(current) = accumulated.as_mut() {
                            for (sum, output) in current.iter_mut().zip(outputs) {
                                *sum = compile_encoding_binary_template(
                                    builder,
                                    &compiler,
                                    MatrixBinaryOp::Add,
                                    sum,
                                    &output,
                                    gate_id,
                                )?;
                            }
                        } else {
                            accumulated = Some(outputs);
                        }
                    }
                    let outputs =
                        accumulated.ok_or(CircuitCompileError::InvalidArity { gate: gate_id })?;
                    insert_call_outputs(&mut values, &info.output_gate_ids, outputs, gate_id)?;
                    continue;
                }
            };
            values.insert(gate_id, value);
        }
        if supplied.next().is_some() {
            return Err(CircuitCompileError::ExtraInputs);
        }
        collect_outputs(circuit, &values)
    }
}

fn compile_public_key_binary_template(
    builder: &mut GraphBuilder,
    compiler: &BggPublicKeyCompiler,
    operation: MatrixBinaryOp,
    lhs: &BggPublicKeyWire,
    rhs: &BggPublicKeyWire,
) -> Result<BggPublicKeyWire, CircuitCompileError> {
    let name = match operation {
        MatrixBinaryOp::Add => "bgg-public-key-add",
        MatrixBinaryOp::Subtract => "bgg-public-key-sub",
        MatrixBinaryOp::Multiply => "bgg-public-key-mul",
    };
    let mut template = GraphBuilder::new(name, Vec::new());
    let template_lhs =
        BggPublicKeyWire { matrix: template.input("lhs", lhs.matrix.matrix_type.clone()) };
    let template_rhs =
        BggPublicKeyWire { matrix: template.input("rhs", rhs.matrix.matrix_type.clone()) };
    let output = match operation {
        MatrixBinaryOp::Add => compiler.add(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Subtract => compiler.sub(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Multiply => compiler.mul(&mut template, &template_lhs, &template_rhs),
    };
    template.output("0_matrix", &output.matrix);
    let mut outputs = builder.subgraph_call(
        template.finish(),
        vec![lhs.matrix.wire, rhs.matrix.wire],
        &[output.matrix.matrix_type],
    )?;
    Ok(BggPublicKeyWire { matrix: outputs.remove(0) })
}

fn compile_encoding_binary_template(
    builder: &mut GraphBuilder,
    compiler: &BggEncodingCompiler,
    operation: MatrixBinaryOp,
    lhs: &BggEncodingWire,
    rhs: &BggEncodingWire,
    gate: usize,
) -> Result<BggEncodingWire, CircuitCompileError> {
    let plaintext_kind = match (&lhs.plaintext, &rhs.plaintext) {
        (Some(_), Some(_)) => "revealed",
        (None, None) => "hidden",
        _ => {
            return Err(CircuitCompileError::Encoding {
                gate,
                source: crate::encoding::EncodingCompileError::PlaintextMismatch,
            });
        }
    };
    let operation_name = match operation {
        MatrixBinaryOp::Add => "add",
        MatrixBinaryOp::Subtract => "sub",
        MatrixBinaryOp::Multiply => "mul",
    };
    let mut template =
        GraphBuilder::new(format!("bgg-encoding-{operation_name}-{plaintext_kind}"), Vec::new());
    let template_lhs = BggEncodingWire {
        vector: template.input("0_lhs_vector", lhs.vector.matrix_type.clone()),
        pubkey: BggPublicKeyWire {
            matrix: template.input("1_lhs_pubkey", lhs.pubkey.matrix.matrix_type.clone()),
        },
        plaintext: lhs
            .plaintext
            .as_ref()
            .map(|plaintext| template.input("2_lhs_plaintext", plaintext.matrix_type.clone())),
    };
    let template_rhs = BggEncodingWire {
        vector: template.input("3_rhs_vector", rhs.vector.matrix_type.clone()),
        pubkey: BggPublicKeyWire {
            matrix: template.input("4_rhs_pubkey", rhs.pubkey.matrix.matrix_type.clone()),
        },
        plaintext: rhs
            .plaintext
            .as_ref()
            .map(|plaintext| template.input("5_rhs_plaintext", plaintext.matrix_type.clone())),
    };
    let output = match operation {
        MatrixBinaryOp::Add => compiler.add(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Subtract => compiler.sub(&mut template, &template_lhs, &template_rhs),
        MatrixBinaryOp::Multiply => compiler.mul(&mut template, &template_lhs, &template_rhs),
    }
    .map_err(|source| CircuitCompileError::Encoding { gate, source })?;
    template.output("0_vector", &output.vector);
    template.output("1_pubkey", &output.pubkey.matrix);
    if let Some(plaintext) = &output.plaintext {
        template.output("2_plaintext", plaintext);
    }
    let mut args = vec![lhs.vector.wire, lhs.pubkey.matrix.wire];
    if let Some(plaintext) = &lhs.plaintext {
        args.push(plaintext.wire);
    }
    args.extend([rhs.vector.wire, rhs.pubkey.matrix.wire]);
    if let Some(plaintext) = &rhs.plaintext {
        args.push(plaintext.wire);
    }
    let mut output_types =
        vec![output.vector.matrix_type.clone(), output.pubkey.matrix.matrix_type.clone()];
    if let Some(plaintext) = &output.plaintext {
        output_types.push(plaintext.matrix_type.clone());
    }
    let mut outputs = builder.subgraph_call(template.finish(), args, &output_types)?;
    let vector = outputs.remove(0);
    let pubkey = BggPublicKeyWire { matrix: outputs.remove(0) };
    let plaintext = (!outputs.is_empty()).then(|| outputs.remove(0));
    Ok(BggEncodingWire { vector, pubkey, plaintext })
}

fn compile_public_key_scalar_template(
    builder: &mut GraphBuilder,
    compiler: &BggPublicKeyCompiler,
    input: &BggPublicKeyWire,
    scalar: &crate::MatrixWire,
    large: bool,
) -> Result<BggPublicKeyWire, CircuitCompileError> {
    let name = if large { "bgg-public-key-large-scalar" } else { "bgg-public-key-small-scalar" };
    let mut template = GraphBuilder::new(name, Vec::new());
    let template_input =
        BggPublicKeyWire { matrix: template.input("input", input.matrix.matrix_type.clone()) };
    let template_scalar = template.input("scalar", scalar.matrix_type.clone());
    let output = if large {
        compiler.large_scalar_mul(&mut template, &template_input, &template_scalar)
    } else {
        compiler.small_scalar_mul(&mut template, &template_input, &template_scalar)
    };
    template.output("0_matrix", &output.matrix);
    let mut outputs = builder.subgraph_call(
        template.finish(),
        vec![input.matrix.wire, scalar.wire],
        &[output.matrix.matrix_type],
    )?;
    Ok(BggPublicKeyWire { matrix: outputs.remove(0) })
}

fn compile_encoding_scalar_template(
    builder: &mut GraphBuilder,
    compiler: &BggEncodingCompiler,
    input: &BggEncodingWire,
    scalar: &crate::MatrixWire,
    large: bool,
) -> Result<BggEncodingWire, CircuitCompileError> {
    let plaintext_kind = if input.plaintext.is_some() { "revealed" } else { "hidden" };
    let operation = if large { "large" } else { "small" };
    let mut template =
        GraphBuilder::new(format!("bgg-encoding-{operation}-scalar-{plaintext_kind}"), Vec::new());
    let template_input = BggEncodingWire {
        vector: template.input("0_vector", input.vector.matrix_type.clone()),
        pubkey: BggPublicKeyWire {
            matrix: template.input("1_pubkey", input.pubkey.matrix.matrix_type.clone()),
        },
        plaintext: input
            .plaintext
            .as_ref()
            .map(|plaintext| template.input("2_plaintext", plaintext.matrix_type.clone())),
    };
    let template_scalar = template.input("3_scalar", scalar.matrix_type.clone());
    let output = if large {
        compiler.large_scalar_mul(&mut template, &template_input, &template_scalar)
    } else {
        compiler.small_scalar_mul(&mut template, &template_input, &template_scalar)
    };
    template.output("0_vector", &output.vector);
    template.output("1_pubkey", &output.pubkey.matrix);
    if let Some(plaintext) = &output.plaintext {
        template.output("2_plaintext", plaintext);
    }
    let mut args = vec![input.vector.wire, input.pubkey.matrix.wire];
    if let Some(plaintext) = &input.plaintext {
        args.push(plaintext.wire);
    }
    args.push(scalar.wire);
    let mut output_types =
        vec![output.vector.matrix_type.clone(), output.pubkey.matrix.matrix_type.clone()];
    if let Some(plaintext) = &output.plaintext {
        output_types.push(plaintext.matrix_type.clone());
    }
    let mut outputs = builder.subgraph_call(template.finish(), args, &output_types)?;
    let vector = outputs.remove(0);
    let pubkey = BggPublicKeyWire { matrix: outputs.remove(0) };
    let plaintext = (!outputs.is_empty()).then(|| outputs.remove(0));
    Ok(BggEncodingWire { vector, pubkey, plaintext })
}

fn scalar_u32_bound(
    builder: &mut GraphBuilder,
    source: &GateParamSource<Vec<u32>>,
    bindings: &[SubCircuitParamValue],
    ambient: &mxx_graph_ir::types::MatrixType,
) -> crate::MatrixWire {
    builder.constant_polynomial(
        scalar_type(ambient),
        source.resolve_small_scalar(bindings).iter().map(|value| BigInt::from(*value)),
    )
}

fn scalar_biguint_bound(
    builder: &mut GraphBuilder,
    source: &GateParamSource<Vec<num_bigint::BigUint>>,
    bindings: &[SubCircuitParamValue],
    ambient: &mxx_graph_ir::types::MatrixType,
) -> crate::MatrixWire {
    builder.constant_polynomial(
        scalar_type(ambient),
        source.resolve_large_scalar(bindings).iter().map(|value| BigInt::from(value.clone())),
    )
}

fn lookup_many<T: Clone>(
    values: &BTreeMap<usize, T>,
    gate: &mxx_gadgets::circuit::PolyGate,
) -> Result<Vec<T>, CircuitCompileError> {
    gate.input_gates
        .iter()
        .map(|input| {
            values.get(&input.index()).cloned().ok_or(CircuitCompileError::MissingInput {
                gate: gate.gate_id.index(),
                input: input.index(),
            })
        })
        .collect()
}

fn flatten_inputs<T: Clone>(
    values: &BTreeMap<usize, T>,
    batches: &[mxx_gadgets::circuit::BatchedWire],
    gate: usize,
) -> Result<Vec<T>, CircuitCompileError> {
    batches
        .iter()
        .flat_map(|batch| batch.gate_ids())
        .map(|input| {
            values
                .get(&input.index())
                .cloned()
                .ok_or(CircuitCompileError::MissingInput { gate, input: input.index() })
        })
        .collect()
}

fn insert_call_outputs<T>(
    values: &mut BTreeMap<usize, T>,
    output_ids: &[GateId],
    outputs: Vec<T>,
    gate: usize,
) -> Result<(), CircuitCompileError> {
    if output_ids.len() != outputs.len() {
        return Err(CircuitCompileError::InvalidArity { gate });
    }
    values.extend(output_ids.iter().map(|id| id.index()).zip(outputs));
    Ok(())
}

fn lookup_unary<'a, T>(
    values: &'a BTreeMap<usize, T>,
    gate: &mxx_gadgets::circuit::PolyGate,
) -> Result<&'a T, CircuitCompileError> {
    let [input] = gate.input_gates.as_slice() else {
        return Err(CircuitCompileError::InvalidArity { gate: gate.gate_id.index() });
    };
    values.get(&input.index()).ok_or(CircuitCompileError::MissingInput {
        gate: gate.gate_id.index(),
        input: input.index(),
    })
}

fn scalar_u32(
    builder: &mut GraphBuilder,
    gate: usize,
    source: &GateParamSource<Vec<u32>>,
    ambient: &mxx_graph_ir::types::MatrixType,
) -> Result<crate::MatrixWire, CircuitCompileError> {
    let GateParamSource::Const(coefficients) = source else {
        return Err(CircuitCompileError::MissingGateContext {
            gate,
            kind: "parameterized small scalar multiplication",
        });
    };
    Ok(builder.constant_polynomial(
        scalar_type(ambient),
        coefficients.iter().map(|value| BigInt::from(*value)),
    ))
}

fn scalar_biguint(
    builder: &mut GraphBuilder,
    gate: usize,
    source: &GateParamSource<Vec<num_bigint::BigUint>>,
    ambient: &mxx_graph_ir::types::MatrixType,
) -> Result<crate::MatrixWire, CircuitCompileError> {
    let GateParamSource::Const(coefficients) = source else {
        return Err(CircuitCompileError::MissingGateContext {
            gate,
            kind: "parameterized large scalar multiplication",
        });
    };
    Ok(builder.constant_polynomial(
        scalar_type(ambient),
        coefficients.iter().map(|value| BigInt::from(value.clone())),
    ))
}

fn scalar_type(ambient: &mxx_graph_ir::types::MatrixType) -> mxx_graph_ir::types::MatrixType {
    mxx_graph_ir::types::MatrixType {
        modulus: ambient.modulus.clone(),
        ring_dimension: ambient.ring_dimension.clone(),
        rows: mxx_graph_ir::IntExpr::constant(1),
        columns: mxx_graph_ir::IntExpr::constant(1),
    }
}

fn lookup_binary<'a, T>(
    values: &'a BTreeMap<usize, T>,
    gate: &mxx_gadgets::circuit::PolyGate,
) -> Result<[&'a T; 2], CircuitCompileError> {
    let [lhs, rhs] = gate.input_gates.as_slice() else {
        return Err(CircuitCompileError::InvalidArity { gate: gate.gate_id.index() });
    };
    Ok([
        values.get(&lhs.index()).ok_or(CircuitCompileError::MissingInput {
            gate: gate.gate_id.index(),
            input: lhs.index(),
        })?,
        values.get(&rhs.index()).ok_or(CircuitCompileError::MissingInput {
            gate: gate.gate_id.index(),
            input: rhs.index(),
        })?,
    ])
}

fn collect_outputs<P: Poly, T: Clone>(
    circuit: &PolyCircuit<P>,
    values: &BTreeMap<usize, T>,
) -> Result<Vec<T>, CircuitCompileError> {
    circuit
        .output_gate_ids()
        .iter()
        .map(|id| {
            values
                .get(&id.index())
                .cloned()
                .ok_or(CircuitCompileError::MissingInput { gate: id.index(), input: id.index() })
        })
        .collect()
}

fn gate_kind_name(kind: &PolyGateType) -> &'static str {
    match kind {
        PolyGateType::Input => "input",
        PolyGateType::Add => "add",
        PolyGateType::Sub => "sub",
        PolyGateType::Mul => "mul",
        PolyGateType::SmallScalarMul { .. } => "small scalar multiplication",
        PolyGateType::LargeScalarMul { .. } => "large scalar multiplication",
        PolyGateType::SlotTransfer { .. } => "slot transfer",
        PolyGateType::SlotReduce { .. } => "slot reduction",
        PolyGateType::PubLut { .. } => "public lookup",
        PolyGateType::SubCircuitOutput { .. } => "sub-circuit call",
        PolyGateType::SummedSubCircuitOutput { .. } => "summed sub-circuit call",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_gadgets::bgg::{encoding::BggEncoding, public_key::BggPublicKey};
    use mxx_graph_ir::{IntExpr, ParamEnv, types::MatrixType};
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::Sign;

    struct NoAdvancedGates;

    impl<W> AdvancedGateLowering<DCRTPoly, W> for NoAdvancedGates {
        fn slot_transfer(
            &mut self,
            _builder: &mut GraphBuilder,
            _input: &W,
            _source_slots: &[(u32, Option<u32>)],
            gate: GateId,
        ) -> Result<W, CircuitCompileError> {
            Err(CircuitCompileError::MissingGateContext {
                gate: gate.index(),
                kind: "slot transfer",
            })
        }

        fn slot_reduce(
            &mut self,
            _builder: &mut GraphBuilder,
            _inputs: &[W],
            _slot_count: usize,
            gate: GateId,
        ) -> Result<W, CircuitCompileError> {
            Err(CircuitCompileError::MissingGateContext {
                gate: gate.index(),
                kind: "slot reduction",
            })
        }

        fn public_lookup(
            &mut self,
            _builder: &mut GraphBuilder,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            _input: &W,
            gate: GateId,
        ) -> Result<W, CircuitCompileError> {
            Err(CircuitCompileError::MissingGateContext {
                gate: gate.index(),
                kind: "public lookup",
            })
        }
    }

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn dcrt_matrix_type(parameters: &DCRTPolyParams, rows: usize, columns: usize) -> MatrixType {
        let modulus: std::sync::Arc<num_bigint::BigUint> = parameters.modulus().into();
        MatrixType {
            modulus: IntExpr::constant(BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn polynomial_row(
        parameters: &DCRTPolyParams,
        columns: usize,
        offset: usize,
    ) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec(
            parameters,
            vec![
                (0..columns)
                    .map(|index| DCRTPoly::const_rotate_poly(parameters, index + offset))
                    .collect(),
            ],
        )
    }

    #[test]
    fn recursively_lowers_registered_subcircuits() {
        let mut child = PolyCircuit::<DCRTPoly>::new();
        let child_inputs = child.input(2).to_vec();
        let child_output = child.add_gate(child_inputs[0], child_inputs[1]);
        child.output([child_output]);

        let mut parent = PolyCircuit::<DCRTPoly>::new();
        let parent_inputs = parent.input(2).to_vec();
        let child_id = parent.register_sub_circuit(child);
        let outputs = parent.call_sub_circuit(child_id, parent_inputs);
        parent.output(outputs);

        let mut builder = GraphBuilder::new("subcircuit", Vec::new());
        let one = BggPublicKeyWire { matrix: builder.input("one", matrix_type(2, 10)) };
        let inputs = [
            BggPublicKeyWire { matrix: builder.input("left", matrix_type(2, 10)) },
            BggPublicKeyWire { matrix: builder.input("right", matrix_type(2, 10)) },
        ];
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let output = compiler
            .compile_public_keys_with_lowering(
                &mut builder,
                &parent,
                one,
                inputs,
                &mut NoAdvancedGates,
            )
            .expect("sub-circuit should lower");
        assert_eq!(output.len(), 1);
        builder.output("result", &output[0].matrix);
        let graph = builder.finish();
        assert!(matches!(
            graph.nodes.last().expect("add node").kind,
            mxx_graph_ir::node::NodeKind::SubgraphCall(_)
        ));
        let add_template = graph.subgraphs.get("bgg-public-key-add").expect("shared add template");
        assert!(add_template.nodes.iter().any(|node| matches!(
            node.kind,
            mxx_graph_ir::node::NodeKind::MatrixBinary(mxx_graph_ir::node::MatrixBinaryOp::Add)
        )));
    }

    #[test]
    fn repeated_gate_kind_reuses_one_registered_subgraph_template() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2).to_vec();
        let first = circuit.add_gate(inputs[0], inputs[1]);
        let second = circuit.add_gate(first, inputs[0]);
        circuit.output([second]);

        let mut builder = GraphBuilder::new("template-reuse", Vec::new());
        let one = BggPublicKeyWire { matrix: builder.input("one", matrix_type(2, 10)) };
        let supplied = [
            BggPublicKeyWire { matrix: builder.input("left", matrix_type(2, 10)) },
            BggPublicKeyWire { matrix: builder.input("right", matrix_type(2, 10)) },
        ];
        let compiler = PolyCircuitCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(2),
                decomposed_type: matrix_type(10, 10),
            },
        };
        let output =
            compiler.compile_public_keys(&mut builder, &circuit, one, supplied).expect("circuit");
        builder.output("result", &output[0].matrix);
        let graph = builder.finish();
        assert_eq!(graph.subgraphs.len(), 1);
        assert!(graph.subgraphs.contains_key("bgg-public-key-add"));
        assert_eq!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(node.kind, mxx_graph_ir::node::NodeKind::SubgraphCall(_)))
                .count(),
            2
        );
    }

    #[test]
    fn encoding_multiplication_template_matches_the_concrete_evaluable_operation() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let columns = parameters.modulus_digits();
        let row_type = dcrt_matrix_type(&parameters, 1, columns);
        let scalar_type = dcrt_matrix_type(&parameters, 1, 1);
        let decomposed_type = dcrt_matrix_type(&parameters, columns, columns);
        let mut builder = GraphBuilder::new("encoding-template-conformance", Vec::new());
        let lhs = BggEncodingWire {
            vector: builder.input("lhs_vector", row_type.clone()),
            pubkey: BggPublicKeyWire { matrix: builder.input("lhs_pubkey", row_type.clone()) },
            plaintext: Some(builder.input("lhs_plaintext", scalar_type.clone())),
        };
        let rhs = BggEncodingWire {
            vector: builder.input("rhs_vector", row_type.clone()),
            pubkey: BggPublicKeyWire { matrix: builder.input("rhs_pubkey", row_type.clone()) },
            plaintext: Some(builder.input("rhs_plaintext", scalar_type)),
        };
        let compiler = BggEncodingCompiler {
            public_key: BggPublicKeyCompiler {
                base: IntExpr::constant(BigInt::from(1u64 << parameters.base_bits())),
                decomposed_type,
            },
        };
        let output = compile_encoding_binary_template(
            &mut builder,
            &compiler,
            MatrixBinaryOp::Multiply,
            &lhs,
            &rhs,
            0,
        )
        .expect("multiplication template");
        builder.output("vector", &output.vector);
        builder.output("pubkey", &output.pubkey.matrix);
        builder.output("plaintext", output.plaintext.as_ref().expect("revealed plaintext"));
        let graph = builder.finish();
        assert_eq!(graph.subgraphs.len(), 1);
        let elaborated = mxx_graph_ir::validate(&graph, &ParamEnv::default()).expect("validation");

        let lhs_vector = polynomial_row(&parameters, columns, 0);
        let lhs_pubkey = polynomial_row(&parameters, columns, 1);
        let rhs_vector = polynomial_row(&parameters, columns, 2);
        let rhs_pubkey = polynomial_row(&parameters, columns, 3);
        let lhs_plaintext = DCRTPoly::const_rotate_poly(&parameters, 1);
        let rhs_plaintext = DCRTPoly::const_rotate_poly(&parameters, 2);
        let expected = BggEncoding::new(
            lhs_vector.clone(),
            BggPublicKey::new(lhs_pubkey.clone(), true),
            Some(lhs_plaintext.clone()),
        ) * &BggEncoding::new(
            rhs_vector.clone(),
            BggPublicKey::new(rhs_pubkey.clone(), true),
            Some(rhs_plaintext.clone()),
        );
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let result = execute(
            &elaborated,
            &mut backend,
            BTreeMap::from([
                ("lhs_vector".to_owned(), RuntimeValue::Matrix(lhs_vector)),
                ("lhs_pubkey".to_owned(), RuntimeValue::Matrix(lhs_pubkey)),
                (
                    "lhs_plaintext".to_owned(),
                    RuntimeValue::Matrix(DCRTPolyMatrix::from_poly_vec(
                        &parameters,
                        vec![vec![lhs_plaintext]],
                    )),
                ),
                ("rhs_vector".to_owned(), RuntimeValue::Matrix(rhs_vector)),
                ("rhs_pubkey".to_owned(), RuntimeValue::Matrix(rhs_pubkey)),
                (
                    "rhs_plaintext".to_owned(),
                    RuntimeValue::Matrix(DCRTPolyMatrix::from_poly_vec(
                        &parameters,
                        vec![vec![rhs_plaintext]],
                    )),
                ),
            ]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("execution");
        let RuntimeValue::Matrix(vector) = &result.outputs["vector"] else {
            panic!("vector output")
        };
        let RuntimeValue::Matrix(pubkey) = &result.outputs["pubkey"] else {
            panic!("pubkey output")
        };
        let RuntimeValue::Matrix(plaintext) = &result.outputs["plaintext"] else {
            panic!("plaintext output")
        };
        assert_eq!(vector, &expected.vector);
        assert_eq!(pubkey, &expected.pubkey.matrix);
        assert_eq!(plaintext.entry(0, 0), expected.plaintext.expect("plaintext"));
    }
}
