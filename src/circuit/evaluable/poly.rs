#[cfg(feature = "gpu")]
use crate::poly::PolyParams;
use crate::{circuit::evaluable::Evaluable, poly::Poly};
use num_bigint::BigUint;

impl<P: Poly> Evaluable for P {
    type Params = P::Params;
    type P = P;
    type Compact = Vec<u8>;

    fn to_compact(self) -> Self::Compact {
        self.to_compact_bytes()
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self::from_compact_bytes(params, compact)
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        params.params_for_device(device_id)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        self.clone() * Self::from_u32s(params, scalar)
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        self.clone() * Self::from_biguints(params, scalar)
    }
}
