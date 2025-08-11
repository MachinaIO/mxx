use crate::{
    circuit::gate::GateId,
    lookup::{PltEvaluator, PublicLut},
    poly::Poly,
};

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}

impl<P: Poly> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(&self, _: &P::Params, plt: &PublicLut<P>, input: P, _: GateId) -> P {
        // Lookup returns (k, y_k); we just return y_k for polynomial evaluation.
        plt.f.get(&input).expect("PolyPltEvaluator's public lookup cannot fetch y_k").1.clone()
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
