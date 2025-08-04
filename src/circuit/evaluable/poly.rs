use crate::{
    circuit::evaluable::Evaluable,
    element::PolyElem,
    lookup::public_lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use rayon::prelude::*;
use std::fmt::Debug;

impl<P: Poly> Evaluable for P {
    type Params = P::Params;
    type P = P;

    fn rotate(self, params: &Self::Params, shift: usize) -> Self {
        let mut coeffs = self.coeffs();
        coeffs.rotate_right(shift);
        Self::from_coeffs(params, &coeffs)
    }

    fn from_digits(params: &Self::Params, _: &Self, digits: &[u32]) -> Self {
        let coeffs: Vec<P::Elem> = digits
            .par_iter()
            .map(|&digit| <P::Elem as PolyElem>::constant(&params.modulus(), digit as u64))
            .collect();
        Self::from_coeffs(params, &coeffs)
    }
}

pub trait PltEvaluator<E: Evaluable>: Send + Sync {
    fn public_lookup(&self, params: &E::Params, plt: &PublicLut<E::P>, input: E, id: usize) -> E;
}

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}
impl<P: Poly> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(&self, _: &P::Params, plt: &PublicLut<P>, input: P, _: usize) -> P {
        plt.f
            .get(&input)
            .expect("PolyPltEvaluator's public lookup cannot fetch y_k")
            .1
            .clone()
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
