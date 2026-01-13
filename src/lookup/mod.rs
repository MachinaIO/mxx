pub mod ggh15_eval;
pub mod lwe_eval;
pub mod poly;

use crate::{circuit::gate::GateId, poly::Poly};
use std::{fmt, sync::Arc};

pub trait PltEvaluator<E: crate::circuit::evaluable::Evaluable>: Send + Sync {
    fn public_lookup(
        &self,
        params: &E::Params,
        plt: &PublicLut<E::P>,
        one: E,
        input: E,
        gate_id: GateId,
        lut_id: usize,
    ) -> E;
}

#[derive(Clone)]
pub struct PublicLut<P: Poly> {
    f: Arc<dyn Fn(&P::Params, &P) -> Option<(usize, P)> + Send + Sync>,
    len: usize,
    max_output_row: (usize, <P as Poly>::Elem),
}

impl<P: Poly + 'static> PublicLut<P> {
    pub fn new<F>(
        params: &P::Params,
        len: usize,
        f: F,
        max_output_row: Option<(usize, P::Elem)>,
    ) -> Self
    where
        F: Fn(&P::Params, &P) -> Option<(usize, P)> + Send + Sync + 'static,
    {
        assert!(len > 0, "f must contain at least one element");
        let f: Arc<dyn Fn(&P::Params, &P) -> Option<(usize, P)> + Send + Sync> = Arc::new(f);
        let max_output_row =
            max_output_row.unwrap_or_else(|| Self::compute_max_output_row(params, f.as_ref(), len));
        Self { f, len, max_output_row }
    }

    /// Builds a LUT over inputs in the range [0, len), using `to_const_int` on the input
    /// polynomial to recover the index. Pass `max_output_row` to avoid scanning all entries.
    pub fn new_from_usize_range<F>(
        params: &P::Params,
        len: usize,
        f: F,
        max_output_row: Option<(usize, P::Elem)>,
    ) -> Self
    where
        F: Fn(&P::Params, usize) -> (usize, P) + Send + Sync + 'static,
    {
        let f = Arc::new(f);
        let lookup = {
            let f = Arc::clone(&f);
            move |params: &P::Params, x: &P| {
                let idx = x.to_const_int();
                if idx >= len {
                    return None;
                }
                Some((f)(params, idx))
            }
        };
        Self::new(params, len, lookup, max_output_row)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, params: &P::Params, x: &P) -> Option<(usize, P)> {
        (self.f)(params, x)
    }

    pub fn entries<'a>(
        &'a self,
        params: &'a P::Params,
    ) -> Box<dyn Iterator<Item = (P, (usize, P))> + Send + 'a> {
        Box::new((0..self.len).map(move |idx| {
            let input = P::from_usize_to_constant(params, idx);
            let (k, y) = (self.f)(params, &input)
                .unwrap_or_else(|| panic!("LUT entry {} missing from 0..len range", idx));
            (input, (k, y))
        }))
    }

    pub fn max_output_row(&self) -> &(usize, <P as Poly>::Elem) {
        &self.max_output_row
    }

    fn compute_max_output_row(
        params: &P::Params,
        f: &dyn Fn(&P::Params, &P) -> Option<(usize, P)>,
        len: usize,
    ) -> (usize, P::Elem) {
        (0..len)
            .filter_map(|idx| {
                let input = P::from_usize_to_constant(params, idx);
                let (k, y_k) = f(params, &input)
                    .unwrap_or_else(|| panic!("LUT entry {} missing from 0..len range", idx));
                y_k.coeffs().into_iter().max().map(|coeff| (k, coeff))
            })
            .max_by(|a, b| a.1.cmp(&b.1))
            .expect("no coefficients found in any y_k")
    }
}

impl<P: Poly> fmt::Debug for PublicLut<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublicLut")
            .field("len", &self.len)
            .field("max_output_row", &self.max_output_row)
            .finish()
    }
}
