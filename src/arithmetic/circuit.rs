use num_bigint::BigUint;

use crate::{
    circuit::PolyCircuit,
    gadgets::{
        crt::{CrtContext, CrtPoly},
        isolate::IsolationGadget,
    },
    poly::Poly,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct ArithmeticCircuit<P: Poly> {
    pub limb_bit_size: usize,
    pub num_crt_limbs: usize,
    pub packed_limbs: usize,
    pub num_inputs: usize,
    pub poly_circuit: PolyCircuit<P>,
    pub ctx: Arc<CrtContext<P>>,
    pub use_reconstruction: bool,
    //inputs + intermediate results
    pub all_values: Vec<CrtPoly<P>>,
}

impl<P: Poly> ArithmeticCircuit<P> {
    pub fn to_poly_circuit(&mut self, isolate_gadget: IsolationGadget<P>) {
        let input_gates = self.poly_circuit.input(self.num_inputs);
        let mut decomposed_outputs = Vec::with_capacity(input_gates.len() * self.packed_limbs);
        for g in input_gates {
            let isolated = isolate_gadget.isolate_terms(&mut self.poly_circuit, g);
            decomposed_outputs.extend(isolated);
        }
        self.poly_circuit.output(decomposed_outputs);
    }

    pub fn setup(
        params: &<P as Poly>::Params,
        num_crt_limbs: usize,
        limb_bit_size: usize,
        packed_limbs: usize,
        inputs: &[BigUint],
        use_reconstruction: bool,
    ) -> Self {
        let num_inputs = inputs.len();
        let mut poly_circuit = PolyCircuit::<P>::new();
        let ctx = Arc::new(CrtContext::setup(&mut poly_circuit, &params, limb_bit_size));
        let mut crt_inputs = Vec::with_capacity(num_inputs);
        for _ in 0..num_inputs {
            let crt_poly = CrtPoly::input(ctx.clone(), &mut poly_circuit);
            crt_inputs.push(crt_poly);
        }

        ArithmeticCircuit {
            limb_bit_size,
            num_crt_limbs,
            packed_limbs,
            num_inputs,
            poly_circuit,
            ctx,
            use_reconstruction,
            all_values: crt_inputs,
        }
    }

    /// rhs + lhs
    pub fn add(&mut self, rhs_index: usize, lhs_index: usize) -> usize {
        assert!(rhs_index < self.all_values.len(), "rhs_index out of bounds");
        assert!(lhs_index < self.all_values.len(), "lhs_index out of bounds");

        let rhs_crt = &self.all_values[rhs_index];
        let lhs_crt = &self.all_values[lhs_index];
        let result_crt = rhs_crt.add(lhs_crt, &mut self.poly_circuit);

        self.all_values.push(result_crt);
        self.all_values.len() - 1
    }

    /// rhs - lhs
    pub fn sub(&mut self, rhs_index: usize, lhs_index: usize) -> usize {
        assert!(rhs_index < self.all_values.len(), "rhs_index out of bounds");
        assert!(lhs_index < self.all_values.len(), "lhs_index out of bounds");

        let rhs_crt = &self.all_values[rhs_index];
        let lhs_crt = &self.all_values[lhs_index];
        let result_crt = rhs_crt.sub(lhs_crt, &mut self.poly_circuit);

        self.all_values.push(result_crt);
        self.all_values.len() - 1
    }

    /// rhs * lhs
    pub fn mul(&mut self, rhs_index: usize, lhs_index: usize) -> usize {
        assert!(rhs_index < self.all_values.len(), "rhs_index out of bounds");
        assert!(lhs_index < self.all_values.len(), "lhs_index out of bounds");

        let rhs_crt = &self.all_values[rhs_index];
        let lhs_crt = &self.all_values[lhs_index];
        let result_crt = rhs_crt.mul(lhs_crt, &mut self.poly_circuit);

        self.all_values.push(result_crt);
        self.all_values.len() - 1
    }

    /// Finalize a value at the given index and set it as output
    pub fn finalize(&mut self, value_index: usize) {
        assert!(value_index < self.all_values.len(), "value_index out of bounds");

        let result_crt = &self.all_values[value_index];
        if self.use_reconstruction {
            let output_gate = result_crt.finalize_reconst(&mut self.poly_circuit);
            self.poly_circuit.output(vec![output_gate]);
        } else {
            // Output all CRT slots as specified in the spec
            let gates = result_crt.finalize_crt(&mut self.poly_circuit);
            self.poly_circuit.output(gates);
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        PolyParams,
        dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
    };
    use num_bigint::BigUint;
    use num_traits::ToPrimitive;

    #[test]
    fn test_arithmetic_circuit_operations() {
        let params = DCRTPolyParams::default();
        let (moduli, _, _) = params.to_crt();
        let large_a = BigUint::from(140000u64);
        let large_b = BigUint::from(132000u64);
        let large_c = BigUint::from(50000u64);

        // Expected results for each operation.
        let add_expected = &large_a + &large_b;
        let mul_expected = &large_a * &large_c;
        let sub_expected = &large_a - &large_c;

        // Verify modular arithmetic correctness for all operations.
        for (i, &qi) in moduli.iter().enumerate() {
            let a_mod_qi = (&large_a % qi as u64).to_u64().unwrap();
            let b_mod_qi = (&large_b % qi as u64).to_u64().unwrap();
            let c_mod_qi = (&large_c % qi as u64).to_u64().unwrap();

            let add_mod_qi = (a_mod_qi + b_mod_qi) % qi as u64;
            let mul_mod_qi = (a_mod_qi * c_mod_qi) % qi as u64;
            let sub_mod_qi = if a_mod_qi >= c_mod_qi {
                a_mod_qi - c_mod_qi
            } else {
                qi as u64 - (c_mod_qi - a_mod_qi)
            };

            let add_expected_mod_qi = (&add_expected % qi as u64).to_u64().unwrap();
            let mul_expected_mod_qi = (&mul_expected % qi as u64).to_u64().unwrap();
            let sub_expected_mod_qi = (&sub_expected % qi as u64).to_u64().unwrap();

            assert_eq!(
                add_mod_qi, add_expected_mod_qi,
                "Addition should be consistent in slot {}",
                i
            );
            assert_eq!(
                mul_mod_qi, mul_expected_mod_qi,
                "Multiplication should be consistent in slot {}",
                i
            );
            assert_eq!(
                sub_mod_qi, sub_expected_mod_qi,
                "Subtraction should be consistent in slot {}",
                i
            );
        }

        let inputs = vec![large_a.clone(), large_b.clone(), large_c.clone()];
        let (_, crt_bits, _) = params.to_crt();
        let limb_bit_size = 5;

        // Test mixed operations in single circuit: (a + b) * c - a.
        let mut mixed_circuit = ArithmeticCircuit::<DCRTPoly>::setup(
            &params,
            crt_bits.div_ceil(limb_bit_size),
            limb_bit_size,
            1,
            &inputs,
            true,
        );
        let add_idx = mixed_circuit.add(0, 1); // a + b
        let mul_idx = mixed_circuit.mul(add_idx, 2); // (a + b) * c
        let final_idx = mixed_circuit.sub(mul_idx, 0); // (a + b) * c - a
        mixed_circuit.finalize(final_idx);
        let mixed_result = mixed_circuit.evaluate(&params, &inputs)[0];
        let mixed_expected = ((&large_a + &large_b) * &large_c) - &large_a;
        assert_eq!(
            mixed_result,
            mixed_expected.to_u64().unwrap() as usize,
            "Mixed operations should be correct"
        );
    }
}
