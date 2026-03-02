use crate::{
    agr16::{encoding::Agr16Encoding, public_key::Agr16PublicKey},
    circuit::evaluable::Evaluable,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct Agr16PublicKeyCompact<M: PolyMatrix> {
    pub matrix_bytes: Vec<u8>,
    pub c_times_s_pubkeys_bytes: Vec<Vec<u8>>,
    pub s_power_pubkeys_bytes: Vec<Vec<u8>>,
    pub reveal_plaintext: bool,
    pub _m: PhantomData<M>,
}

#[derive(Debug, Clone)]
pub struct Agr16EncodingCompact<M: PolyMatrix> {
    pub vector_bytes: Vec<u8>,
    pub c_times_s_encodings_bytes: Vec<Vec<u8>>,
    pub s_power_encodings_bytes: Vec<Vec<u8>>,
    pub pubkey: Agr16PublicKeyCompact<M>,
    pub plaintext_bytes: Option<Vec<u8>>,
    pub _m: PhantomData<M>,
}

impl<M: PolyMatrix> Evaluable for Agr16PublicKey<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;
    type Compact = Agr16PublicKeyCompact<M>;

    fn to_compact(self) -> Self::Compact {
        Agr16PublicKeyCompact::<M> {
            matrix_bytes: self.matrix.into_compact_bytes(),
            c_times_s_pubkeys_bytes: self
                .c_times_s_pubkeys
                .into_iter()
                .map(|level| level.into_compact_bytes())
                .collect(),
            s_power_pubkeys_bytes: self
                .s_power_pubkeys
                .into_iter()
                .map(|level| level.into_compact_bytes())
                .collect(),
            reveal_plaintext: self.reveal_plaintext,
            _m: PhantomData,
        }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Agr16PublicKey {
            matrix: M::from_compact_bytes(params, &compact.matrix_bytes),
            c_times_s_pubkeys: compact
                .c_times_s_pubkeys_bytes
                .iter()
                .map(|level_bytes| M::from_compact_bytes(params, level_bytes))
                .collect(),
            s_power_pubkeys: compact
                .s_power_pubkeys_bytes
                .iter()
                .map(|level_bytes| M::from_compact_bytes(params, level_bytes))
                .collect(),
            reveal_plaintext: compact.reveal_plaintext,
        }
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        params.params_for_device(device_id)
    }

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self {
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.unsigned_abs() as usize
        };
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        Self {
            matrix: self.matrix.clone() * &rotate_poly,
            c_times_s_pubkeys: self
                .c_times_s_pubkeys
                .iter()
                .map(|level| level.clone() * &rotate_poly)
                .collect(),
            s_power_pubkeys: self.s_power_pubkeys.clone(),
            reveal_plaintext: self.reveal_plaintext,
        }
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        let scalar = Self::P::from_u32s(params, scalar);
        Self {
            matrix: self.matrix.clone() * &scalar,
            c_times_s_pubkeys: self
                .c_times_s_pubkeys
                .iter()
                .map(|level| level.clone() * &scalar)
                .collect(),
            s_power_pubkeys: self.s_power_pubkeys.clone(),
            reveal_plaintext: self.reveal_plaintext,
        }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[num_bigint::BigUint]) -> Self {
        let scalar = Self::P::from_biguints(params, scalar);
        let row_size = self.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * &scalar;
        Self {
            matrix: self.matrix.mul_decompose(&scalar_gadget),
            c_times_s_pubkeys: self
                .c_times_s_pubkeys
                .iter()
                .map(|level| level.mul_decompose(&scalar_gadget))
                .collect(),
            s_power_pubkeys: self.s_power_pubkeys.clone(),
            reveal_plaintext: self.reveal_plaintext,
        }
    }
}

impl<M: PolyMatrix> Evaluable for Agr16Encoding<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;
    type Compact = Agr16EncodingCompact<M>;

    fn to_compact(self) -> Self::Compact {
        Agr16EncodingCompact::<M> {
            vector_bytes: self.vector.into_compact_bytes(),
            c_times_s_encodings_bytes: self
                .c_times_s_encodings
                .into_iter()
                .map(|level| level.into_compact_bytes())
                .collect(),
            s_power_encodings_bytes: self
                .s_power_encodings
                .into_iter()
                .map(|level| level.into_compact_bytes())
                .collect(),
            pubkey: self.pubkey.to_compact(),
            plaintext_bytes: self.plaintext.map(|p| p.to_compact_bytes()),
            _m: PhantomData,
        }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Agr16Encoding {
            vector: M::from_compact_bytes(params, &compact.vector_bytes),
            c_times_s_encodings: compact
                .c_times_s_encodings_bytes
                .iter()
                .map(|level_bytes| M::from_compact_bytes(params, level_bytes))
                .collect(),
            s_power_encodings: compact
                .s_power_encodings_bytes
                .iter()
                .map(|level_bytes| M::from_compact_bytes(params, level_bytes))
                .collect(),
            pubkey: Agr16PublicKey::from_compact(params, &compact.pubkey),
            plaintext: compact
                .plaintext_bytes
                .as_ref()
                .map(|bytes| M::P::from_compact_bytes(params, bytes)),
        }
    }

    #[cfg(feature = "gpu")]
    fn params_for_eval_device(params: &Self::Params, device_id: i32) -> Self::Params {
        params.params_for_device(device_id)
    }

    fn rotate(&self, params: &Self::Params, shift: i32) -> Self {
        let pubkey = self.pubkey.rotate(params, shift);
        let shift = if shift >= 0 {
            shift as usize
        } else {
            params.ring_dimension() as usize - shift.unsigned_abs() as usize
        };
        let rotate_poly = <M::P>::const_rotate_poly(params, shift);
        Self {
            vector: self.vector.clone() * &rotate_poly,
            c_times_s_encodings: self
                .c_times_s_encodings
                .iter()
                .map(|level| level.clone() * &rotate_poly)
                .collect(),
            s_power_encodings: self.s_power_encodings.clone(),
            pubkey,
            plaintext: self.plaintext.clone().map(|p| p * &rotate_poly),
        }
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        let scalar_poly = Self::P::from_u32s(params, scalar);
        Self {
            vector: self.vector.clone() * &scalar_poly,
            c_times_s_encodings: self
                .c_times_s_encodings
                .iter()
                .map(|level| level.clone() * &scalar_poly)
                .collect(),
            s_power_encodings: self.s_power_encodings.clone(),
            pubkey: self.pubkey.small_scalar_mul(params, scalar),
            plaintext: self.plaintext.clone().map(|p| p * &scalar_poly),
        }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[num_bigint::BigUint]) -> Self {
        let scalar_poly = Self::P::from_biguints(params, scalar);
        let row_size = self.pubkey.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * &scalar_poly;
        Self {
            vector: self.vector.mul_decompose(&scalar_gadget),
            c_times_s_encodings: self
                .c_times_s_encodings
                .iter()
                .map(|level| level.mul_decompose(&scalar_gadget))
                .collect(),
            s_power_encodings: self.s_power_encodings.clone(),
            pubkey: self.pubkey.large_scalar_mul(params, scalar),
            plaintext: self.plaintext.clone().map(|p| p * &scalar_poly),
        }
    }
}
