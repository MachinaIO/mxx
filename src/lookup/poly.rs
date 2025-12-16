use crate::{
    circuit::gate::GateId,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
};

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}

impl<P: Poly> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(&self, params: &P::Params, plt: &PublicLut<P>, input: P, _: GateId) -> P {
        match plt.get(params, &input) {
            Some(outputs) => outputs.1,
            None => {
                panic!("output of the lookup evaluation not found; input: {:?}", input);
            }
        }
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
