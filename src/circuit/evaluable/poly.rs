use crate::{
    circuit::{evaluable::Evaluable, gate::GateId},
    element::PolyElem,
    lookup::public_lookup::PublicLut,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
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

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        let fin_ring_coeffs: Vec<P::Elem> = scalar
            .iter()
            .map(|coeff| <P::Elem as PolyElem>::new(coeff.clone(), params.modulus()))
            .collect();
        self.clone() * Self::from_coeffs(params, &fin_ring_coeffs)
    }
}

pub trait PltEvaluator<E: Evaluable>: Send + Sync {
    fn public_lookup(&self, params: &E::Params, plt: &PublicLut<E::P>, input: E, id: GateId) -> E;
}

#[derive(Debug, Clone)]
pub struct PolyPltEvaluator {}
impl<P: Poly> PltEvaluator<P> for PolyPltEvaluator {
    fn public_lookup(&self, _: &P::Params, plt: &PublicLut<P>, input: P, _: GateId) -> P {
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
