use crate::{
    circuit::gate::GateId,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
};

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}

impl<P: Poly> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(
        &self,
        params: &P::Params,
        plt: &PublicLut<P>,
        input: P,
        gate_id: GateId,
        _: usize,
    ) -> P {
        // Input is assumed to be a constant polynomial; use its constant coefficient as the key.
        let const_coeff = input
            .coeffs()
            .first()
            .cloned()
            .expect("input polynomial must contain at least one coefficient");

        let output_coeff = match plt.get(params, &const_coeff) {
            Some(outputs) => outputs.1,
            None => panic!(
                "output of the lookup evaluation not found; gate_id: {:?}, input: {:?}",
                gate_id, input
            ),
        };

        P::from_elem_to_constant(params, &output_coeff)
    }
}

impl Default for PolyPltEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl PolyPltEvaluator {
    pub fn new() -> Self {
        Self {}
    }
}
