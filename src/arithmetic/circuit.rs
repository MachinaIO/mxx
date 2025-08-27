use crate::{
    circuit::PolyCircuit,
    gadgets::{
        crt::{CrtContext, CrtPoly},
        isolate::IsolationGadget,
    },
    poly::Poly,
};
use num_bigint::BigUint;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArithGateId(usize);

impl ArithGateId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    pub fn as_usize(&self) -> usize {
        self.0
    }
}

impl From<usize> for ArithGateId {
    fn from(id: usize) -> Self {
        Self(id)
    }
}

impl From<ArithGateId> for usize {
    fn from(id: ArithGateId) -> Self {
        id.0
    }
}

#[derive(Clone)]
pub struct ArithmeticCircuit<P: Poly> {
    pub limb_bit_size: usize,
    pub num_limbs: usize,
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
        num_limbs: usize,
        limb_bit_size: usize,
        packed_limbs: usize,
        inputs: &[BigUint],
        use_reconstruction: bool,
    ) -> Self {
        let num_inputs = inputs.len();
        let mut poly_circuit = PolyCircuit::<P>::new();
        let ctx = Arc::new(CrtContext::setup(&mut poly_circuit, params, limb_bit_size));
        let mut crt_inputs = Vec::with_capacity(num_inputs);
        for _ in 0..num_inputs {
            let crt_poly = CrtPoly::input(ctx.clone(), &mut poly_circuit);
            crt_inputs.push(crt_poly);
        }

        ArithmeticCircuit {
            limb_bit_size,
            num_limbs,
            packed_limbs,
            num_inputs,
            poly_circuit,
            ctx,
            use_reconstruction,
            all_values: crt_inputs,
        }
    }

    /// lhs + rhs
    pub fn add(&mut self, lhs_index: ArithGateId, rhs_index: ArithGateId) -> ArithGateId {
        let rhs_idx = rhs_index.as_usize();
        let lhs_idx = lhs_index.as_usize();
        assert!(rhs_idx < self.all_values.len(), "rhs_index out of bounds");
        assert!(lhs_idx < self.all_values.len(), "lhs_index out of bounds");

        let lhs_crt = &self.all_values[lhs_idx];
        let rhs_crt = &self.all_values[rhs_idx];
        let result_crt = lhs_crt.add(rhs_crt, &mut self.poly_circuit);

        self.all_values.push(result_crt);
        ArithGateId::new(self.all_values.len() - 1)
    }

    /// lhs - rhs
    pub fn sub(&mut self, lhs_index: ArithGateId, rhs_index: ArithGateId) -> ArithGateId {
        let rhs_idx = rhs_index.as_usize();
        let lhs_idx = lhs_index.as_usize();
        assert!(rhs_idx < self.all_values.len(), "rhs_index out of bounds");
        assert!(lhs_idx < self.all_values.len(), "lhs_index out of bounds");

        let lhs_crt = &self.all_values[lhs_idx];
        let rhs_crt = &self.all_values[rhs_idx];
        let result_crt = lhs_crt.sub(rhs_crt, &mut self.poly_circuit);

        self.all_values.push(result_crt);
        ArithGateId::new(self.all_values.len() - 1)
    }

    /// lhs * rhs
    pub fn mul(&mut self, lhs_index: ArithGateId, rhs_index: ArithGateId) -> ArithGateId {
        let rhs_idx = rhs_index.as_usize();
        let lhs_idx = lhs_index.as_usize();
        assert!(rhs_idx < self.all_values.len(), "rhs_index out of bounds");
        assert!(lhs_idx < self.all_values.len(), "lhs_index out of bounds");

        let lhs_crt = &self.all_values[lhs_idx];
        let rhs_crt = &self.all_values[rhs_idx];

        let result_crt = lhs_crt.mul(rhs_crt, &mut self.poly_circuit);

        self.all_values.push(result_crt);
        ArithGateId::new(self.all_values.len() - 1)
    }

    /// Output a value at the given index and set it as output
    pub fn output(&mut self, value_index: ArithGateId) {
        let idx = value_index.as_usize();
        assert!(idx < self.all_values.len(), "value_index out of bounds");
        let result_crt = &self.all_values[idx];
        if self.use_reconstruction {
            let output_gate = result_crt.finalize_reconst(&mut self.poly_circuit);
            self.poly_circuit.output(vec![output_gate]);
        } else {
            let gates = result_crt.finalize_crt(&mut self.poly_circuit);
            self.poly_circuit.output(gates);
        };
    }
}
