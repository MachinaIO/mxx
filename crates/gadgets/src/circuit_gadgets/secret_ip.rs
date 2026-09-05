use crate::{
    circuit::{PolyCircuit, gate::GateId},
    poly::Poly,
};

pub fn secret_inner_product<P: Poly>(
    circuit: &mut PolyCircuit<P>,
    public_vec: &[GateId],
    secret_vec: &[GateId],
) -> GateId {
    assert_eq!(public_vec.len(), secret_vec.len(), "vector lengths must match");
    if public_vec.is_empty() {
        return circuit.const_zero_gate().as_single_wire();
    }

    // Multiply with public input on the left to keep BGG encoding semantics.
    let mut acc = circuit.mul_gate(public_vec[0], secret_vec[0]);
    for (&public_id, &secret_id) in public_vec.iter().zip(secret_vec.iter()).skip(1) {
        let prod = circuit.mul_gate(public_id, secret_id);
        acc = circuit.add_gate(acc, prod);
    }
    acc.as_single_wire()
}

#[cfg(test)]
mod tests {
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
            Poly as ConcretePoly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, BigUint};
    use std::{collections::BTreeMap, convert::Infallible};

    struct MatrixLowering {
        ring: Ring,
    }

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
            Ok(match operation {
                PolyGateKind::Add => lhs.clone() + rhs.clone(),
                PolyGateKind::Sub => lhs.clone() - rhs.clone(),
                PolyGateKind::Mul => lhs.clone() * rhs.clone(),
                _ => unreachable!("binary lowering receives only arithmetic operations"),
            })
        }

        fn small_scalar_mul(
            &mut self,
            input: &Mat,
            scalar: &[u32],
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            Ok(input.clone() * self.ring.polynomial(scalar.iter().copied().map(IntExpr::constant)))
        }

        fn large_scalar_mul(
            &mut self,
            input: &Mat,
            scalar: &[BigUint],
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            Ok(input.clone() *
                self.ring
                    .polynomial(scalar.iter().cloned().map(BigInt::from).map(IntExpr::constant)))
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

        fn slot_anchor_reduce(
            &mut self,
            input: &Mat,
            _num_blocks: u32,
            _lane_scalars: &[num_bigint::BigUint],
            _gate: GateInstance<'_>,
        ) -> Result<Mat, Infallible> {
            Ok(input.clone())
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
    fn runtime_result_matches_the_primitive_inner_product() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let ring = Ring::new(
            BigInt::from(parameters.modulus().as_ref().clone()),
            parameters.ring_dimension() as usize,
        );
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(4).to_vec();
        let output = secret_inner_product(&mut circuit, &inputs[..2], &inputs[2..]);
        circuit.output([output]);
        let input_wires =
            (0..4).map(|index| ring.input(format!("input-{index}"), (1, 1))).collect::<Vec<_>>();
        let outputs = lower_circuit(
            &circuit,
            ring.polynomial([IntExpr::constant(1)]),
            input_wires,
            &mut MatrixLowering { ring: ring.clone() },
        )
        .expect("matrix lowering is infallible");
        let graph = DslContext::new("secret-inner-product-runtime")
            .output("output", outputs[0].clone())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let values = [2usize, 3, 5, 7].map(|value| {
            DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_usize_to_constant(&parameters, value)],
            )
        });
        let result = execute(
            &graph,
            &mut cpu_backend([parameters]),
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    (format!("input-{index}"), RuntimeValue::matrix(value.clone()))
                })
                .collect::<BTreeMap<_, _>>(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .expect("runtime execution");
        let RuntimeValue::Matrix(actual) = &result.outputs["output"] else {
            panic!("inner-product output must be a matrix")
        };
        assert_eq!(
            actual.as_ref(),
            &(values[0].clone() * values[2].entry(0, 0) +
                values[1].clone() * values[3].entry(0, 0))
        );
    }
}
