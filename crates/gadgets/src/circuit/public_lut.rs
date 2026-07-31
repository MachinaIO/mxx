use crate::poly::Poly;
use std::{fmt, sync::Arc};

/// BGG-independent public lookup-table descriptor stored by [`super::PolyCircuit`].
#[derive(Clone)]
pub struct PublicLut<P: Poly> {
    f: Arc<dyn Fn(&P::Params, u64) -> Option<(u64, P::Elem)> + Send + Sync>,
    len: u64,
    max_output_row: (u64, P::Elem),
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
        Self { f: Arc::new(f), len, max_output_row }
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
        Box::new((0..self.len).map(move |input| {
            let (row, output) = (self.f)(params, input)
                .unwrap_or_else(|| panic!("LUT entry {input} missing from 0..len range"));
            (input, (row, output))
        }))
    }

    pub fn max_output_row(&self) -> &(u64, P::Elem) {
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
                    .unwrap_or_else(|| panic!("LUT entry {input} missing from 0..len range"))
            })
            .max_by(|a, b| a.1.cmp(&b.1))
            .expect("a public lookup table must contain at least one output")
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
