use crate::{
    arithmetic::circuit::ArithmeticCircuit, gadgets::crt::biguint_to_crt_poly,
    lookup::poly::PolyPltEvaluator, poly::Poly,
};
use num_bigint::BigUint;

impl<P: Poly> ArithmeticCircuit<P> {
    pub fn evaluate(&self, params: &P::Params, inputs: &[BigUint]) -> Vec<usize> {
        let one = P::const_one(params);
        let mut all_input_polys = Vec::new();

        for input in inputs {
            let crt_limbs = biguint_to_crt_poly(self.limb_bit_size, params, input);
            all_input_polys.extend(crt_limbs);
        }

        let plt_evaluator = PolyPltEvaluator::new();
        let results = self.poly_circuit.eval(params, &one, &all_input_polys, Some(plt_evaluator));

        if self.use_reconstruction {
            // With reconstruction, only one result
            vec![results[0].to_const_int()]
        } else {
            // Without reconstruction, all CRT slots
            results.iter().map(|r| r.to_const_int()).collect()
        }
    }
}
