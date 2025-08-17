use crate::{
    circuit::gate::GateId,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
};

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}

impl<P: Poly> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(&self, params: &P::Params, plt: &PublicLut<P>, input: P, _: GateId) -> P {
        let outputs = plt.get(params, &input).expect("output of the lookup evaluation not found");
        outputs.1
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
