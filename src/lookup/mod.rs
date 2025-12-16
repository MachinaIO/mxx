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
    pub f: HashMap<<P as Poly>::Elem, (usize, <P as Poly>::Elem)>, /* the `i`-th hashmap in
                                                                    * `Vec` corresponds
                                                                    * the lookup table for the
                                                                    * `i`-th
                                                                    * coefficient */
    pub max_output_row: (usize, <P as Poly>::Elem),
}

impl<P: Poly> PublicLut<P> {
    pub fn new(f: HashMap<<P as Poly>::Elem, (usize, <P as Poly>::Elem)>) -> Self {
        assert!(!f.is_empty(), "f must contain at least one element");
        let max_output_row = f
            .par_iter()
            .map(|(_, (k, y_k))| (*k, y_k.clone()))
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

    pub fn get(&self, _: &P::Params, x: &<P as Poly>::Elem) -> Option<(usize, <P as Poly>::Elem)> {
        self.f.get(x).cloned()
    }

    pub fn max_output_row(&self) -> &(usize, <P as Poly>::Elem) {
        &self.max_output_row
    }
}
