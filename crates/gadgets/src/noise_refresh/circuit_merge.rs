//! Merge circuits for decoded noise-refresh material.
//!
//! The merge phase starts after Ring-GSW decryption has already produced slotwise polynomial wires.
//! It combines decoded error and mask wires by ordinary circuit addition.  Keeping this phase
//! separate lets tests and benchmarks reuse one decoded mask wire many times without repeatedly
//! evaluating the expensive Ring-GSW decrypt path.

use crate::{
    circuit::{PolyCircuit, gate::GateId},
    poly::{Poly, PolyParams},
};

/// Builds a merge subcircuit for `value_count` decoded error/mask pairs.
///
/// Inputs are ordered as all decoded error wires followed by all decoded mask wires.  Output `i`
/// is `decoded_errors[i] + decoded_masks[i]`.
pub fn build_refreshed_wire_merge_subcircuit<P: Poly>(value_count: usize) -> PolyCircuit<P> {
    assert!(value_count > 0, "value_count must be positive");
    let mut circuit = PolyCircuit::<P>::new();
    // Every decoded error or mask is passed as its own one-wire input. The
    // `at(0)` below selects the sole wire of each input object; `value_count`
    // counts how many such inputs exist, not slots inside a batched input.
    let decoded_errors =
        (0..value_count).map(|_| circuit.input(1).at(0).as_single_wire()).collect::<Vec<GateId>>();
    let decoded_masks =
        (0..value_count).map(|_| circuit.input(1).at(0).as_single_wire()).collect::<Vec<GateId>>();
    let outputs = decoded_errors
        .iter()
        .zip(decoded_masks.iter())
        .map(|(&error, &mask)| circuit.add_gate(error, mask).as_single_wire())
        .collect::<Vec<_>>();
    circuit.output(outputs);
    circuit
}

/// Builds the all-CRT merge circuit for one gadget digit of one refreshed wire.
///
/// The CRT depth is read from `params.to_crt()`.  The resulting circuit accepts one decoded error
/// wire per CRT level and one decoded mask wire per CRT level, then emits the element-wise sums.
pub fn build_refreshed_wire_digit_all_crt_merge<P>(params: &P::Params) -> PolyCircuit<P>
where
    P: Poly,
{
    let (_q_moduli, _crt_bits, crt_depth) = params.to_crt();
    build_refreshed_wire_merge_subcircuit::<P>(crt_depth)
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::circuit::{GateInstance, GraphCircuitLowering, PolyGateKind, lower_circuit};
    use mxx_ir_core::{
        GraphBuilder, IntExpr, MatrixWire, ParamEnv,
        node::{ConstantMatrix, MatrixBinaryOp},
        types::MatrixType,
        validate,
    };
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;
    use num_bigint::BigUint;
    use std::convert::Infallible;

    struct MatrixLowering;

    impl GraphCircuitLowering<DCRTPoly> for MatrixLowering {
        type Wire = MatrixWire;
        type Error = Infallible;

        fn binary(
            &mut self,
            builder: &mut GraphBuilder,
            operation: PolyGateKind,
            lhs: &MatrixWire,
            rhs: &MatrixWire,
            _gate: GateInstance<'_>,
        ) -> Result<MatrixWire, Infallible> {
            let operation = match operation {
                PolyGateKind::Add => MatrixBinaryOp::Add,
                PolyGateKind::Sub => MatrixBinaryOp::Subtract,
                PolyGateKind::Mul => MatrixBinaryOp::Multiply,
                _ => unreachable!("merge circuit contains only addition"),
            };
            Ok(builder.matrix_binary(operation, lhs, rhs, lhs.matrix_type.clone()))
        }

        fn small_scalar_mul(
            &mut self,
            _builder: &mut GraphBuilder,
            input: &MatrixWire,
            _scalar: &[u32],
            _gate: GateInstance<'_>,
        ) -> Result<MatrixWire, Infallible> {
            Ok(input.clone())
        }

        fn large_scalar_mul(
            &mut self,
            _builder: &mut GraphBuilder,
            input: &MatrixWire,
            _scalar: &[BigUint],
            _gate: GateInstance<'_>,
        ) -> Result<MatrixWire, Infallible> {
            Ok(input.clone())
        }

        fn slot_transfer(
            &mut self,
            _builder: &mut GraphBuilder,
            input: &MatrixWire,
            _source_slots: &[(u32, Option<u32>)],
            _gate: GateInstance<'_>,
        ) -> Result<MatrixWire, Infallible> {
            Ok(input.clone())
        }

        fn slot_reduce(
            &mut self,
            _builder: &mut GraphBuilder,
            inputs: &[MatrixWire],
            _slot_count: usize,
            _gate: GateInstance<'_>,
        ) -> Result<MatrixWire, Infallible> {
            Ok(inputs[0].clone())
        }

        fn public_lookup(
            &mut self,
            _builder: &mut GraphBuilder,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            input: &MatrixWire,
            _gate: GateInstance<'_>,
        ) -> Result<MatrixWire, Infallible> {
            Ok(input.clone())
        }
    }

    #[test]
    fn merge_template_preserves_output_order_and_lowers_to_graph_ir() {
        let circuit = build_refreshed_wire_merge_subcircuit::<DCRTPoly>(3);
        assert_eq!(circuit.output_gate_ids().len(), 3);
        let input_gates = circuit.sorted_input_gate_ids();
        assert_eq!(input_gates.len(), 6);
        for (index, output) in circuit.output_gate_ids().iter().copied().enumerate() {
            let gate = circuit.gate(output);
            assert_eq!(gate.gate_type, crate::circuit::PolyGateType::Add);
            assert_eq!(gate.input_gates, vec![input_gates[index], input_gates[3 + index]]);
        }
        let matrix_type = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let mut builder = GraphBuilder::new("noise-refresh-merge-template", Vec::new());
        let one = builder.constant_matrix(matrix_type.clone(), ConstantMatrix::Identity);
        let inputs = (0..6)
            .map(|index| builder.input(format!("input_{index}"), matrix_type.clone()))
            .collect::<Vec<_>>();
        let outputs = lower_circuit(&mut builder, &circuit, one, inputs, &mut MatrixLowering)
            .expect("merge lowering");
        assert_eq!(outputs.len(), 3);
        for (index, output) in outputs.iter().enumerate() {
            builder.value_output_wire(format!("output_{index}"), output.wire);
        }
        validate(&builder.finish(), &ParamEnv::default()).expect("valid merge Graph IR");
    }
}
