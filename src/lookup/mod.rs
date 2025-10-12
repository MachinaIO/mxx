pub mod poly;
// pub mod simple_eval;
pub mod lwe_eval;

use rayon::prelude::*;

use crate::{circuit::gate::GateId, poly::Poly};
use std::collections::HashMap;

pub trait PltEvaluator<E: crate::circuit::evaluable::Evaluable>: Send + Sync {
    fn public_lookup(&self, params: &E::Params, plt: &PublicLut<E::P>, input: E, id: GateId) -> E;
}

#[derive(Debug, Clone)]
pub struct PublicLut<P: Poly> {
    pub f: HashMap<P, (usize, P)>, /* the `i`-th hashmap in `Vec` corresponds
                                    * the lookup table for the `i`-th
                                    * coefficient */
    pub max_output_row: (usize, <P as Poly>::Elem),
}

impl<P: Poly> PublicLut<P> {
    pub fn new(f: HashMap<P, (usize, P)>) -> Self {
        assert!(!f.is_empty(), "f must contain at least one element");
        let max_output_row = f
            .par_iter()
            .filter_map(|(_, (k, y_k))| y_k.coeffs().iter().max().cloned().map(|coeff| (*k, coeff)))
            .max_by(|a, b| a.1.cmp(&b.1))
            .expect("no coefficients found in any y_k");
        Self { f, max_output_row }
    }
    pub fn len(&self) -> usize {
        self.f.len()
    }

    pub fn is_empty(&self) -> bool {
        self.f.is_empty()
    }

    pub fn get(&self, _: &P::Params, x: &P) -> Option<(usize, P)> {
        self.f.get(x).cloned()
    }

    pub fn max_output_row(&self) -> &(usize, <P as Poly>::Elem) {
        &self.max_output_row
    }
}
