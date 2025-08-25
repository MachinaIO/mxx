pub mod poly;
pub mod simple_eval;

use crate::{
    circuit::gate::GateId,
    element::PolyElem,
    poly::{Poly, PolyParams},
};
use std::collections::HashMap;

pub trait PltEvaluator<E: crate::circuit::evaluable::Evaluable>: Send + Sync {
    fn public_lookup(&self, params: &E::Params, plt: &PublicLut<E::P>, input: E, id: GateId) -> E;
}

#[derive(Debug, Clone, Default)]
pub struct PublicLut<P: Poly> {
    pub fs: Vec<HashMap<P::Elem, (usize, P::Elem)>>, /* the `i`-th hashmap in `Vec` corresponds
                                                      * the lookup table for the `i`-th
                                                      * coefficient */
}

impl<P: Poly> PublicLut<P> {
    pub fn new(fs: Vec<HashMap<P::Elem, (usize, P::Elem)>>) -> Self {
        Self { fs }
    }
    pub fn len(&self) -> usize {
        self.fs.iter().map(|f| f.len()).max().unwrap()
    }

    pub fn get(&self, params: &P::Params, x: &P) -> Option<(Vec<usize>, P)> {
        let in_coeffs = x.coeffs();
        let mut row_idxes = vec![];
        let mut out_coeffs = vec![];
        for (i, f) in self.fs.iter().enumerate() {
            match f.get(&in_coeffs[i]) {
                Some((out_idx, out_coeff)) => {
                    row_idxes.push(*out_idx);
                    out_coeffs.push(out_coeff.clone());
                }
                None => {
                    return None;
                }
            }
        }
        for i in self.fs.len()..in_coeffs.len() {
            assert_eq!(
                &in_coeffs[i],
                &<P::Elem as PolyElem>::zero(&params.modulus()),
                "The {i}-th input {:?} must be zero",
                &in_coeffs[i]
            );
        }
        Some((row_idxes, P::from_coeffs(params, &out_coeffs)))
    }

    pub fn max_output_row(&self) -> (usize, usize, <P as Poly>::Elem) {
        assert!(!self.fs.is_empty(), "f must contain at least one element");

        let mut max_y_k = None;
        let mut max_i = 0;
        let mut max_k = 0;

        for (i, hashmap) in self.fs.iter().enumerate() {
            for (_, (k, y_k)) in hashmap.iter() {
                match &max_y_k {
                    None => {
                        max_y_k = Some(y_k.clone());
                        max_i = i;
                        max_k = *k;
                    }
                    Some(current_max) => {
                        if y_k > current_max {
                            max_y_k = Some(y_k.clone());
                            max_i = i;
                            max_k = *k;
                        }
                    }
                }
            }
        }

        let max_y_k = max_y_k.expect("no elements found in any hashmap");
        (max_i, max_k, max_y_k)
    }
}
