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
    use crate::circuit::{
        ArithmeticCircuitLowering, CircuitLoweringTypes, GateInstance, PolyGateKind,
        PublicLookupLowering, SlotOperationLowering, lower_circuit,
    };
    use mxx_dsl::{DslContext, Mat, Ring};
    use mxx_ir_core::{IntExpr, ParamEnv};
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
    use num_bigint::{BigInt, BigUint};
    use std::{collections::BTreeMap, convert::Infallible};

    struct MatrixLowering;

    impl CircuitLoweringTypes for MatrixLowering {
        type Wire = Mat;
        type Error = Infallible;
    }

    impl ArithmeticCircuitLowering<DCRTPoly> for MatrixLowering {
        fn binary(
            &mut self,
            operation: PolyGateKind,
            lhs: &Mat,
            rhs: &Mat,
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            let output = match operation {
                PolyGateKind::Add => lhs.clone() + rhs.clone(),
                PolyGateKind::Sub => lhs.clone() - rhs.clone(),
                PolyGateKind::Mul => lhs.clone() * rhs.clone(),
                _ => unreachable!("merge circuit contains only addition"),
            };
            Ok(output)
        }

        fn small_scalar_mul(
            &mut self,
            input: &Mat,
            _scalar: &[u32],
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            Ok(input.clone())
        }

        fn large_scalar_mul(
            &mut self,
            input: &Mat,
            _scalar: &[BigUint],
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            Ok(input.clone())
        }
    }

    impl SlotOperationLowering<DCRTPoly> for MatrixLowering {
        fn slot_transfer(
            &mut self,
            input: &Mat,
            _source_slots: &[(u32, Option<u32>)],
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            Ok(input.clone())
        }

        fn slot_reduce(
            &mut self,
            inputs: &[Mat],
            _slot_count: usize,
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            Ok(inputs[0].clone())
        }
    }

    impl PublicLookupLowering<DCRTPoly> for MatrixLowering {
        fn public_lookup(
            &mut self,
            _circuit: &PolyCircuit<DCRTPoly>,
            _lookup_id: usize,
            input: &Mat,
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
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
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let ring = Ring::new(
            IntExpr::constant(BigInt::from(parameters.modulus().as_ref().clone())),
            IntExpr::constant(parameters.ring_dimension()),
        );
        let shape = (1, 1);
        let one = ring.identity(1);
        let inputs =
            (0..6).map(|index| ring.input(format!("input_{index}"), shape)).collect::<Vec<_>>();
        let outputs =
            lower_circuit(&circuit, one, inputs, &mut MatrixLowering).expect("merge lowering");
        assert_eq!(outputs.len(), 3);
        let mut context = DslContext::new("noise-refresh-merge-template");
        for (index, output) in outputs.into_iter().enumerate() {
            context =
                context.public_output(format!("output_{index}"), output).expect("unique output");
        }
        let built = context.build().expect("build merge Graph IR");
        let validated = built.validate(&ParamEnv::default()).expect("valid merge Graph IR");
        let input_values = (0..6)
            .map(|index| {
                DCRTPolyMatrix::from_poly_vec_row(
                    &parameters,
                    vec![DCRTPoly::from_usize_to_constant(&parameters, index + 1)],
                )
            })
            .collect::<Vec<_>>();
        let result = execute(
            &validated,
            &mut cpu_backend([parameters]),
            input_values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    (format!("input_{index}"), RuntimeValue::matrix(value.clone()))
                })
                .collect::<BTreeMap<_, _>>(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("execute merge Graph IR");
        for index in 0..3 {
            let RuntimeValue::Matrix(actual) = &result.outputs[&format!("output_{index}")] else {
                panic!("merge output must be a matrix")
            };
            assert_eq!(actual.as_ref(), &(input_values[index].clone() + &input_values[3 + index]));
        }
    }
}
