use crate::{
    circuit::evaluable::Evaluable,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;

impl<P: Poly> Evaluable for P {
    type Params = P::Params;
    type P = P;

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self {
        let mut coeffs = self.coeffs();
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.unsigned_abs() as usize
        };
        coeffs.rotate_right(shift);
        Self::from_coeffs(params, &coeffs)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        self.clone() * Self::from_u32s(params, scalar)
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        self.clone() * Self::from_biguints(params, scalar)
    }
}
