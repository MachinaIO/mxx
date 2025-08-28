pub mod poly;
// pub mod simple_eval;
pub mod lwe_eval;

use crate::{circuit::gate::GateId, poly::Poly};
use std::collections::HashMap;

pub trait PltEvaluator<E: crate::circuit::evaluable::Evaluable>: Send + Sync {
    fn public_lookup(&self, params: &E::Params, plt: &PublicLut<E::P>, input: E, id: GateId) -> E;
}

#[derive(Debug, Clone, Default)]
pub struct PublicLut<P: Poly> {
    pub f: HashMap<P, (usize, P)>, /* the `i`-th hashmap in `Vec` corresponds
                                    * the lookup table for the `i`-th
                                    * coefficient */
}

impl<P: Poly> PublicLut<P> {
    pub fn new(f: HashMap<P, (usize, P)>) -> Self {
        Self { f }
    }
    pub fn len(&self) -> usize {
        self.f.len()
    }

    pub fn get(&self, _: &P::Params, x: &P) -> Option<(usize, P)> {
        self.f.get(x).cloned()
    }

    pub fn max_output_row(&self) -> (usize, <P as Poly>::Elem) {
        assert!(!self.f.is_empty(), "f must contain at least one element");
        self.f
            .iter()
            .filter_map(|(_, (k, y_k))| y_k.coeffs().iter().max().cloned().map(|coeff| (*k, coeff)))
            .max_by(|a, b| a.1.cmp(&b.1))
            .expect("no coefficients found in any y_k")
    }
}
