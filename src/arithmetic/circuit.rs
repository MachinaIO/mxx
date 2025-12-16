use crate::{
    circuit::PolyCircuit,
    gadgets::arith::basic::{BasicModuloPoly, BasicModuloPolyContext},
    poly::Poly,
    utils::log_mem,
};
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
    pub num_inputs: usize,
    pub poly_circuit: PolyCircuit<P>,
    pub ctx: Arc<BasicModuloPolyContext<P>>,
    // pub use_packing: bool,
    // pub use_reconstruction: bool,
    // inputs + intermediate results
    pub all_values: Vec<BasicModuloPoly<P>>,
}

impl<P: Poly> ArithmeticCircuit<P> {
    pub fn setup(
        params: &<P as Poly>::Params,
        limb_bit_size: usize,
        num_inputs: usize,
        dummy_scalar: bool,
    ) -> Self {
        let mut poly_circuit = PolyCircuit::<P>::new();
        let mut all_values = Vec::with_capacity(num_inputs);
        log_mem("before ModuloPolyContext setup");
        let ctx = Arc::new(BasicModuloPolyContext::setup(
            &mut poly_circuit,
            params,
            limb_bit_size,
            dummy_scalar,
        ));
        log_mem("after ModuloPolyContext setup");
        let crt_polys = (0..num_inputs)
            .map(|_| BasicModuloPoly::input(ctx.clone(), &mut poly_circuit))
            .collect::<Vec<_>>();
        all_values.extend(crt_polys);

        ArithmeticCircuit { limb_bit_size, num_inputs, poly_circuit, ctx, all_values }
    }

    pub fn to_poly_circuit(self) -> PolyCircuit<P> {
        self.poly_circuit
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
        let out = result_crt.finalize(&mut self.poly_circuit);

        self.poly_circuit.output(vec![out]);
    }

    pub fn benchmark_multiplication_tree(
        params: &<P as Poly>::Params,
        limb_bit_size: usize,
        height: usize,
        dummy_scalar: bool,
    ) -> Self {
        assert!(height >= 1, "height must be at least 1 to build a multiplication tree");
        let num_inputs =
            1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");

        let mut circuit = Self::setup(params, limb_bit_size, num_inputs, dummy_scalar);

        // Collect the leaf identifiers representing the primary inputs.
        let mut current_layer: Vec<ArithGateId> = (0..num_inputs).map(ArithGateId::from).collect();

        // Repeatedly pairwise multiply adjacent nodes until a single root remains.
        while current_layer.len() > 1 {
            debug_assert!(current_layer.len().is_multiple_of(2), "layer size must stay even");
            let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
            log_mem(format!("before layer size {}", current_layer.len()));
            for pair in current_layer.chunks(2) {
                let parent = circuit.mul(pair[0], pair[1]);
                next_layer.push(parent);
            }
            log_mem(format!("after layer size {}", current_layer.len()));
            current_layer = next_layer;
        }

        let root = current_layer.pop().expect("multiplication tree must contain at least one node");
        circuit.output(root);
        circuit
    }
}
