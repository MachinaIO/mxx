use crate::{
    bgg::public_key::{BggPublicKey, BggPublicKeyCompact},
    circuit::evaluable::Evaluable,
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
};
use num_bigint::BigUint;
use rayon::prelude::*;
use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

#[derive(Debug, Clone)]
pub struct BggEncoding<M: PolyMatrix> {
    pub vector: M,
    pub pubkey: BggPublicKey<M>,
    pub plaintext: Option<<M as PolyMatrix>::P>,
}

#[derive(Debug, Clone)]
pub struct BggEncodingCompact<M: PolyMatrix> {
    pub vector_bytes: Vec<u8>,
    pub pubkey: BggPublicKeyCompact<M>,
    pub plaintext_bytes: Option<Vec<u8>>,
}

impl<M: PolyMatrix> BggEncoding<M> {
    pub fn new(
        vector: M,
        pubkey: BggPublicKey<M>,
        plaintext: Option<<M as PolyMatrix>::P>,
    ) -> Self {
        Self { vector, pubkey, plaintext }
    }

    pub fn concat_vector(&self, others: &[Self]) -> M {
        self.vector.concat_columns(&others.par_iter().map(|x| &x.vector).collect::<Vec<_>>()[..])
    }

    /// Reads an encoding with id from files under the given directory.
    pub fn read_from_files<P: AsRef<std::path::Path> + Send + Sync>(
        params: &<M::P as Poly>::Params,
        d1: usize,
        log_base_q: usize,
        dir_path: P,
        id: &str,
        reveal_plaintext: bool,
    ) -> Self {
        let ncol = d1 * log_base_q;

        // Read the vector
        let vector = M::read_from_files(params, 1, ncol, &dir_path, &format!("{id}_vector"));

        // Read the pubkey
        let pubkey = BggPublicKey::read_from_files(
            params,
            d1,
            ncol,
            &dir_path,
            &format!("{id}_pubkey"),
            reveal_plaintext,
        );

        // If reveal_plaintext is true, read the plaintext
        let plaintext = if reveal_plaintext {
            Some(M::P::read_from_file(params, &dir_path, &format!("{id}_plaintext")))
        } else {
            None
        };

        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Add for BggEncoding<M> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for BggEncoding<M> {
    type Output = Self;
    fn add(self, other: &Self) -> Self {
        let vector = self.vector + &other.vector;
        let pubkey = self.pubkey + &other.pubkey;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Sub for BggEncoding<M> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for BggEncoding<M> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self {
        let vector = self.vector - &other.vector;
        let pubkey = self.pubkey - &other.pubkey;
        let plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
        Self { vector, pubkey, plaintext }
    }
}

impl<M: PolyMatrix> Mul for BggEncoding<M> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for BggEncoding<M> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self {
        if self.plaintext.is_none() {
            panic!("Unknown plaintext for the left-hand input of multiplication");
        }
        let first_term = self.vector.mul_decompose(&other.pubkey.matrix);
        let second_term = other.vector.clone() * self.plaintext.as_ref().unwrap();
        let new_vector = first_term + second_term;
        let new_plaintext = match (self.plaintext, other.plaintext.as_ref()) {
            (Some(a), Some(b)) => Some(a * b),
            _ => None,
        };

        let new_pubkey = BggPublicKey {
            matrix: self.pubkey.matrix.mul_decompose(&other.pubkey.matrix),
            reveal_plaintext: self.pubkey.reveal_plaintext & other.pubkey.reveal_plaintext,
        };
        Self { vector: new_vector, pubkey: new_pubkey, plaintext: new_plaintext }
    }
}

impl<M: PolyMatrix> Evaluable for BggEncoding<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;
    type Compact = BggEncodingCompact<M>;

    fn to_compact(self) -> Self::Compact {
        BggEncodingCompact {
            vector_bytes: self.vector.into_compact_bytes(),
            pubkey: BggPublicKeyCompact::new(
                self.pubkey.matrix.into_compact_bytes(),
                self.pubkey.reveal_plaintext,
            ),
            plaintext_bytes: self.plaintext.map(|p| p.to_compact_bytes()),
        }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        BggEncoding {
            vector: M::from_compact_bytes(params, &compact.vector_bytes),
            pubkey: BggPublicKey {
                matrix: M::from_compact_bytes(params, &compact.pubkey.matrix_bytes),
                reveal_plaintext: compact.pubkey.reveal_plaintext,
            },
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
        let vector = self.vector.clone() * &rotate_poly;
        let plaintext = self.plaintext.clone().map(|plaintext| plaintext * rotate_poly);
        Self { vector, pubkey, plaintext }
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        let scalar = Self::P::from_u32s(params, scalar);
        let vector = self.vector.clone() * &scalar;
        let pubkey_matrix = self.pubkey.matrix.clone() * &scalar;
        let pubkey = BggPublicKey::new(pubkey_matrix, self.pubkey.reveal_plaintext);
        let plaintext = self.plaintext.clone().map(|p| p * scalar);
        Self { vector, pubkey, plaintext }
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar = Self::P::from_biguints(params, scalar);
        let row_size = self.pubkey.matrix.row_size();
        let scalar_gadget = M::gadget_matrix(params, row_size) * &scalar;
        let vector = self.vector.mul_decompose(&scalar_gadget);
        let pubkey_matrix = self.pubkey.matrix.mul_decompose(&scalar_gadget);
        let pubkey = BggPublicKey::new(pubkey_matrix, self.pubkey.reveal_plaintext);
        let plaintext = self.plaintext.clone().map(|p| p * scalar);
        Self { vector, pubkey, plaintext }
    }
}
