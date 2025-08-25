use crate::{
    circuit::evaluable::Evaluable,
    element::PolyElem,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use rayon::prelude::*;

impl<P: Poly> Evaluable for P {
    type Params = P::Params;
    type P = P;

    fn rotate(self, params: &Self::Params, shift: i32) -> Self {
        let mut coeffs = self.coeffs();
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.abs() as usize
        };
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
