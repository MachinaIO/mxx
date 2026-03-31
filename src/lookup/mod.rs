pub mod commit_eval;
mod ggh15;
pub mod ggh15_eval;
pub mod lwe_eval;
pub mod poly;
pub mod poly_vec;

use crate::{circuit::gate::GateId, poly::Poly};
use std::{fmt, sync::Arc};

pub trait PltEvaluator<E: crate::circuit::evaluable::Evaluable>: Send + Sync {
    fn public_lookup(
        &self,
        params: &E::Params,
        plt: &PublicLut<E::P>,
        one: &E,
        input: &E,
        gate_id: GateId,
        lut_id: usize,
    ) -> E;
}

#[derive(Clone)]
pub struct PublicLut<P: Poly> {
    f: Arc<dyn Fn(&P::Params, u64) -> Option<(u64, P::Elem)> + Send + Sync>,
    len: u64,
    max_output_row: (u64, <P as Poly>::Elem),
}

impl<P: Poly> PublicLut<P> {
    pub fn new<F>(
        params: &P::Params,
        len: u64,
        f: F,
        max_output_row: Option<(u64, P::Elem)>,
    ) -> Self
    where
        F: Fn(&P::Params, u64) -> Option<(u64, P::Elem)> + Send + Sync + 'static,
    {
        let max_output_row =
            max_output_row.unwrap_or_else(|| Self::compute_max_output_row(params, &f, len));
        let f = Arc::new(f);
        Self { f, len, max_output_row }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, params: &P::Params, x: u64) -> Option<(u64, P::Elem)> {
        (self.f)(params, x)
    }

    pub fn entries<'a>(
        &'a self,
        params: &'a P::Params,
    ) -> Box<dyn Iterator<Item = (u64, (u64, P::Elem))> + Send + 'a> {
        Box::new((0..self.len as u64).map(move |input| {
            let (k, y) = (self.f)(params, input)
                .unwrap_or_else(|| panic!("LUT entry {} missing from 0..len range", input));
            (input, (k, y))
        }))
    }

    #[cfg(feature = "gpu")]
    pub fn entries_multi_gpus<'a>(
        &'a self,
        params_by_device: &'a [&'a P::Params],
    ) -> Box<dyn Iterator<Item = (Vec<u64>, (u64, Vec<P::Elem>))> + Send + 'a> {
        assert!(
            !params_by_device.is_empty(),
            "entries_multi_gpus requires at least one device parameter set"
        );
        Box::new((0..self.len).map(move |input_idx| {
            let mut x_by_device = Vec::with_capacity(params_by_device.len());
            let mut y_by_device = Vec::with_capacity(params_by_device.len());
            let mut row_idx: Option<u64> = None;
            for &device_params in params_by_device {
                let (k, y) = (self.f)(device_params, input_idx).unwrap_or_else(|| {
                    panic!(
                        "LUT entry {} missing from 0..len range on one of GPU device params",
                        input_idx
                    )
                });
                if let Some(expected_k) = row_idx {
                    assert_eq!(
                        expected_k, k,
                        "entries_multi_gpus expects consistent output-row idx across devices for input {}",
                        input_idx
                    );
                } else {
                    row_idx = Some(k);
                }
                x_by_device.push(input_idx);
                y_by_device.push(y);
            }
            (x_by_device, (row_idx.expect("row idx must exist"), y_by_device))
        }))
    }

    pub fn max_output_row(&self) -> &(u64, <P as Poly>::Elem) {
        &self.max_output_row
    }

    fn compute_max_output_row(
        params: &P::Params,
        f: &dyn Fn(&P::Params, u64) -> Option<(u64, P::Elem)>,
        len: u64,
    ) -> (u64, P::Elem) {
        (0..len)
            .map(|input| {
                f(params, input)
                    .unwrap_or_else(|| panic!("LUT entry {} missing from 0..len range", input))
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
