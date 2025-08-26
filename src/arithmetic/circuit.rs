use num_bigint::BigUint;
use tracing::info;

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
    // inputs + intermediate results
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
        info!("mul st");
        let result_crt = rhs_crt.mul(lhs_crt, &mut self.poly_circuit);
        info!("mul end");

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
            let gates = result_crt.finalize_crt(&mut self.poly_circuit);
            self.poly_circuit.output(gates);
        };
    }
}
