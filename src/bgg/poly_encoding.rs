#[cfg(feature = "gpu")]
#[path = "poly_encoding_gpu.rs"]
mod gpu;

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
    path::Path,
    sync::Arc,
};

fn effective_slot_parallelism<M: PolyMatrix>(
    params: &<M::P as Poly>::Params,
    num_slots: usize,
) -> usize {
    if num_slots == 0 {
        return 0;
    }

    let requested = crate::env::bgg_poly_encoding_slot_parallelism().max(1);
    #[cfg(feature = "gpu")]
    {
        gpu::effective_slot_parallelism_gpu(params, num_slots, requested)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = params;
        requested.min(num_slots)
    }
}

fn map_slots_with_params<M, T, F>(params: &<M::P as Poly>::Params, num_slots: usize, f: F) -> Vec<T>
where
    M: PolyMatrix,
    T: Send,
    F: Fn(usize, &<M::P as Poly>::Params) -> T + Send + Sync,
{
    if num_slots == 0 {
        return Vec::new();
    }

    let slot_parallelism = effective_slot_parallelism::<M>(params, num_slots);

    #[cfg(feature = "gpu")]
    {
        gpu::map_slots_with_params_gpu(params, num_slots, slot_parallelism, f)
    }

    #[cfg(not(feature = "gpu"))]
    {
        let mut outputs = Vec::with_capacity(num_slots);
        for slot_start in (0..num_slots).step_by(slot_parallelism) {
            let chunk_len = (slot_start + slot_parallelism).min(num_slots) - slot_start;
            let mut chunk_outputs = (0..chunk_len)
                .into_par_iter()
                .map(|offset| f(slot_start + offset, params))
                .collect::<Vec<_>>();
            outputs.append(&mut chunk_outputs);
        }
        outputs
    }
}

fn share_bytes<B>(bytes: Vec<B>) -> Vec<Arc<[u8]>>
where
    B: Into<Arc<[u8]>>,
{
    bytes.into_iter().map(Into::into).collect()
}

fn read_poly_compact_bytes_from_file<P: AsRef<Path> + Send + Sync>(
    dir_path: P,
    id: &str,
) -> Arc<[u8]> {
    let path = dir_path.as_ref().join(format!("{id}.poly"));
    Arc::<[u8]>::from(
        std::fs::read(&path).unwrap_or_else(|_| panic!("Failed to read polynomial file {path:?}")),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BggPolyEncoding<M: PolyMatrix> {
    pub params: <M::P as Poly>::Params,
    pub vector_bytes: Vec<Arc<[u8]>>,
    pub pubkey: BggPublicKey<M>,
    pub plaintext_bytes: Option<Vec<Arc<[u8]>>>,
}

#[derive(Debug, Clone)]
pub struct BggPolyEncodingCompact<M: PolyMatrix> {
    pub vector_bytes: Vec<Arc<[u8]>>,
    pub pubkey: BggPublicKeyCompact<M>,
    pub plaintext_bytes: Option<Vec<Arc<[u8]>>>,
}

impl<M: PolyMatrix> BggPolyEncoding<M> {
    pub fn new<B>(
        params: <M::P as Poly>::Params,
        vector_bytes: Vec<B>,
        pubkey: BggPublicKey<M>,
        plaintext_bytes: Option<Vec<B>>,
    ) -> Self
    where
        B: Into<Arc<[u8]>>,
    {
        let vector_bytes = share_bytes(vector_bytes);
        let plaintext_bytes = plaintext_bytes.map(share_bytes);
        if let Some(plaintext_bytes) = plaintext_bytes.as_ref() {
            assert_eq!(
                plaintext_bytes.len(),
                vector_bytes.len(),
                "BggPolyEncoding::new requires plaintext_bytes.len() == vector_bytes.len()"
            );
        }
        Self { params, vector_bytes, pubkey, plaintext_bytes }
    }

    pub fn num_slots(&self) -> usize {
        self.vector_bytes.len()
    }

    pub fn vector(&self, slot: usize) -> M {
        self.vector_for_params(&self.params, slot)
    }

    pub fn vector_for_params(&self, params: &<M::P as Poly>::Params, slot: usize) -> M {
        M::from_compact_bytes(params, self.vector_bytes[slot].as_ref())
    }

    pub fn plaintext(&self, slot: usize) -> Option<M::P> {
        self.plaintext_for_params(&self.params, slot)
    }

    pub fn plaintext_for_params(
        &self,
        params: &<M::P as Poly>::Params,
        slot: usize,
    ) -> Option<M::P> {
        self.plaintext_bytes
            .as_ref()
            .map(|plaintext_bytes| M::P::from_compact_bytes(params, plaintext_bytes[slot].as_ref()))
    }

    pub fn concat_vector(&self, others: &[Self]) -> Vec<M> {
        for other in others {
            assert_eq!(
                self.num_slots(),
                other.num_slots(),
                "BggPolyEncoding::concat_vector requires matching num_slots"
            );
            assert_eq!(
                self.params, other.params,
                "BggPolyEncoding::concat_vector requires matching params"
            );
        }

        map_slots_with_params::<M, _, _>(&self.params, self.num_slots(), |slot, local_params| {
            let lhs = self.vector_for_params(local_params, slot);
            let rhs = others
                .iter()
                .map(|other| other.vector_for_params(local_params, slot))
                .collect::<Vec<_>>();
            let rhs_refs = rhs.iter().collect::<Vec<_>>();
            let out = lhs.concat_columns(&rhs_refs);
            drop(rhs);
            out
        })
    }

    /// Reads a poly encoding with id from files under the given directory.
    pub fn read_from_files<P: AsRef<std::path::Path> + Send + Sync>(
        params: &<M::P as Poly>::Params,
        d1: usize,
        log_base_q: usize,
        dir_path: P,
        id: &str,
        reveal_plaintext: bool,
        num_slots: usize,
    ) -> Self {
        let ncol = d1 * log_base_q;
        let vectors: Vec<M> = (0..num_slots)
            .into_par_iter()
            .map(|slot| {
                M::read_from_files(params, 1, ncol, &dir_path, &format!("{id}_slot_{slot}_vector"))
            })
            .collect();
        let pubkey = BggPublicKey::read_from_files(
            params,
            d1,
            ncol,
            &dir_path,
            &format!("{id}_pubkey"),
            reveal_plaintext,
        );
        let plaintext_bytes = if reveal_plaintext {
            Some(
                (0..num_slots)
                    .into_par_iter()
                    .map(|slot| {
                        read_poly_compact_bytes_from_file(
                            &dir_path,
                            &format!("{id}_slot_{slot}_plaintext"),
                        )
                    })
                    .collect(),
            )
        } else {
            None
        };

        let vector_bytes = vectors
            .into_iter()
            .map(|vector| Arc::<[u8]>::from(vector.into_compact_bytes()))
            .collect::<Vec<_>>();
        Self::new(params.clone(), vector_bytes, pubkey, plaintext_bytes)
    }
}

impl<M: PolyMatrix> Add for BggPolyEncoding<M> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self + &other
    }
}

impl<M: PolyMatrix> Add<&Self> for BggPolyEncoding<M> {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        let Self { params, vector_bytes, pubkey, plaintext_bytes } = self;
        assert_eq!(
            vector_bytes.len(),
            other.vector_bytes.len(),
            "BggPolyEncoding::add requires matching num_slots"
        );
        assert_eq!(params, other.params, "BggPolyEncoding::add requires matching params");

        let pubkey = pubkey + &other.pubkey;
        let vector_bytes =
            map_slots_with_params::<M, _, _>(&params, vector_bytes.len(), |slot, local_params| {
                let lhs = M::from_compact_bytes(local_params, vector_bytes[slot].as_ref());
                let rhs = M::from_compact_bytes(local_params, other.vector_bytes[slot].as_ref());
                let out = lhs + &rhs;
                drop(rhs);
                Arc::<[u8]>::from(out.into_compact_bytes())
            });
        let plaintext_bytes = match (plaintext_bytes, other.plaintext_bytes.as_ref()) {
            (Some(lhs), Some(rhs)) => {
                Some(map_slots_with_params::<M, _, _>(&params, lhs.len(), |slot, local_params| {
                    let a = M::P::from_compact_bytes(local_params, lhs[slot].as_ref());
                    let b = M::P::from_compact_bytes(local_params, rhs[slot].as_ref());
                    Arc::<[u8]>::from((a + b).to_compact_bytes())
                }))
            }
            _ => None,
        };
        Self::new(params, vector_bytes, pubkey, plaintext_bytes)
    }
}

impl<M: PolyMatrix> Sub for BggPolyEncoding<M> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self - &other
    }
}

impl<M: PolyMatrix> Sub<&Self> for BggPolyEncoding<M> {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        let Self { params, vector_bytes, pubkey, plaintext_bytes } = self;
        assert_eq!(
            vector_bytes.len(),
            other.vector_bytes.len(),
            "BggPolyEncoding::sub requires matching num_slots"
        );
        assert_eq!(params, other.params, "BggPolyEncoding::sub requires matching params");

        let pubkey = pubkey - &other.pubkey;
        let vector_bytes =
            map_slots_with_params::<M, _, _>(&params, vector_bytes.len(), |slot, local_params| {
                let lhs = M::from_compact_bytes(local_params, vector_bytes[slot].as_ref());
                let rhs = M::from_compact_bytes(local_params, other.vector_bytes[slot].as_ref());
                let out = lhs - &rhs;
                drop(rhs);
                Arc::<[u8]>::from(out.into_compact_bytes())
            });
        let plaintext_bytes = match (plaintext_bytes, other.plaintext_bytes.as_ref()) {
            (Some(lhs), Some(rhs)) => {
                Some(map_slots_with_params::<M, _, _>(&params, lhs.len(), |slot, local_params| {
                    let a = M::P::from_compact_bytes(local_params, lhs[slot].as_ref());
                    let b = M::P::from_compact_bytes(local_params, rhs[slot].as_ref());
                    Arc::<[u8]>::from((a - b).to_compact_bytes())
                }))
            }
            _ => None,
        };
        Self::new(params, vector_bytes, pubkey, plaintext_bytes)
    }
}

impl<M: PolyMatrix> Mul for BggPolyEncoding<M> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        self * &other
    }
}

impl<M: PolyMatrix> Mul<&Self> for BggPolyEncoding<M> {
    type Output = Self;

    fn mul(self, other: &Self) -> Self {
        let Self { params, vector_bytes, pubkey, plaintext_bytes } = self;
        assert_eq!(
            vector_bytes.len(),
            other.vector_bytes.len(),
            "BggPolyEncoding::mul requires matching num_slots"
        );
        assert_eq!(params, other.params, "BggPolyEncoding::mul requires matching params");

        let lhs_plaintext_bytes = plaintext_bytes.as_ref().unwrap_or_else(|| {
            panic!("Unknown plaintext for the left-hand input of multiplication")
        });
        let other_pubkey_matrix_bytes = Arc::<[u8]>::from(other.pubkey.matrix.to_compact_bytes());

        let pubkey = pubkey * &other.pubkey;
        let vector_bytes =
            map_slots_with_params::<M, _, _>(&params, vector_bytes.len(), |slot, local_params| {
                let lhs_vector = M::from_compact_bytes(local_params, vector_bytes[slot].as_ref());
                let rhs_vector =
                    M::from_compact_bytes(local_params, other.vector_bytes[slot].as_ref());
                let rhs_pubkey_matrix =
                    M::from_compact_bytes(local_params, other_pubkey_matrix_bytes.as_ref());
                let lhs_plaintext =
                    M::P::from_compact_bytes(local_params, lhs_plaintext_bytes[slot].as_ref());
                let first_term = lhs_vector.mul_decompose(&rhs_pubkey_matrix);
                let second_term = rhs_vector * &lhs_plaintext;
                drop(rhs_pubkey_matrix);
                drop(lhs_plaintext);
                let out = first_term + second_term;
                Arc::<[u8]>::from(out.into_compact_bytes())
            });
        let plaintext_bytes = match (plaintext_bytes, other.plaintext_bytes.as_ref()) {
            (Some(lhs), Some(rhs)) => {
                Some(map_slots_with_params::<M, _, _>(&params, lhs.len(), |slot, local_params| {
                    let a = M::P::from_compact_bytes(local_params, lhs[slot].as_ref());
                    let b = M::P::from_compact_bytes(local_params, rhs[slot].as_ref());
                    Arc::<[u8]>::from((a * b).to_compact_bytes())
                }))
            }
            _ => None,
        };
        Self::new(params, vector_bytes, pubkey, plaintext_bytes)
    }
}

impl<M: PolyMatrix> Evaluable for BggPolyEncoding<M> {
    type Params = <M::P as Poly>::Params;
    type P = M::P;
    type Compact = BggPolyEncodingCompact<M>;

    fn to_compact(self) -> Self::Compact {
        BggPolyEncodingCompact {
            vector_bytes: self.vector_bytes,
            pubkey: BggPublicKeyCompact::new(
                self.pubkey.matrix.into_compact_bytes(),
                self.pubkey.reveal_plaintext,
            ),
            plaintext_bytes: self.plaintext_bytes,
        }
    }

    fn from_compact(params: &Self::Params, compact: &Self::Compact) -> Self {
        Self::new(
            params.clone(),
            compact.vector_bytes.clone(),
            BggPublicKey {
                matrix: M::from_compact_bytes(params, &compact.pubkey.matrix_bytes),
                reveal_plaintext: compact.pubkey.reveal_plaintext,
            },
            compact.plaintext_bytes.clone(),
        )
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
        let rotate_poly_bytes =
            Arc::<[u8]>::from(<M::P>::const_rotate_poly(params, shift).to_compact_bytes());
        let vector_bytes =
            map_slots_with_params::<M, _, _>(params, self.num_slots(), |slot, local_params| {
                let vector = M::from_compact_bytes(local_params, self.vector_bytes[slot].as_ref());
                let rotate_poly =
                    M::P::from_compact_bytes(local_params, rotate_poly_bytes.as_ref());
                let out = vector * &rotate_poly;
                drop(rotate_poly);
                Arc::<[u8]>::from(out.into_compact_bytes())
            });
        let plaintext_bytes = self.plaintext_bytes.as_ref().map(|plaintext_bytes| {
            map_slots_with_params::<M, _, _>(params, plaintext_bytes.len(), |slot, local_params| {
                let plaintext =
                    M::P::from_compact_bytes(local_params, plaintext_bytes[slot].as_ref());
                let rotate_poly =
                    M::P::from_compact_bytes(local_params, rotate_poly_bytes.as_ref());
                Arc::<[u8]>::from((plaintext * rotate_poly).to_compact_bytes())
            })
        });
        Self::new(params.clone(), vector_bytes, pubkey, plaintext_bytes)
    }

    fn small_scalar_mul(&self, params: &Self::Params, scalar: &[u32]) -> Self {
        let scalar_bytes = Arc::<[u8]>::from(Self::P::from_u32s(params, scalar).to_compact_bytes());
        let vector_bytes =
            map_slots_with_params::<M, _, _>(params, self.num_slots(), |slot, local_params| {
                let vector = M::from_compact_bytes(local_params, self.vector_bytes[slot].as_ref());
                let scalar_poly = M::P::from_compact_bytes(local_params, scalar_bytes.as_ref());
                let out = vector * &scalar_poly;
                drop(scalar_poly);
                Arc::<[u8]>::from(out.into_compact_bytes())
            });
        let plaintext_bytes = self.plaintext_bytes.as_ref().map(|plaintext_bytes| {
            map_slots_with_params::<M, _, _>(params, plaintext_bytes.len(), |slot, local_params| {
                let plaintext =
                    M::P::from_compact_bytes(local_params, plaintext_bytes[slot].as_ref());
                let scalar_poly = M::P::from_compact_bytes(local_params, scalar_bytes.as_ref());
                Arc::<[u8]>::from((plaintext * scalar_poly).to_compact_bytes())
            })
        });
        Self::new(
            params.clone(),
            vector_bytes,
            self.pubkey.small_scalar_mul(params, scalar),
            plaintext_bytes,
        )
    }

    fn large_scalar_mul(&self, params: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_poly = Self::P::from_biguints(params, scalar);
        let row_size = self.pubkey.matrix.row_size();
        let scalar_gadget_bytes = Arc::<[u8]>::from(
            (M::gadget_matrix(params, row_size) * &scalar_poly).into_compact_bytes(),
        );
        let scalar_bytes = Arc::<[u8]>::from(scalar_poly.to_compact_bytes());
        let vector_bytes =
            map_slots_with_params::<M, _, _>(params, self.num_slots(), |slot, local_params| {
                let vector = M::from_compact_bytes(local_params, self.vector_bytes[slot].as_ref());
                let scalar_gadget =
                    M::from_compact_bytes(local_params, scalar_gadget_bytes.as_ref());
                let out = vector.mul_decompose(&scalar_gadget);
                drop(scalar_gadget);
                Arc::<[u8]>::from(out.into_compact_bytes())
            });
        let plaintext_bytes = self.plaintext_bytes.as_ref().map(|plaintext_bytes| {
            map_slots_with_params::<M, _, _>(params, plaintext_bytes.len(), |slot, local_params| {
                let plaintext =
                    M::P::from_compact_bytes(local_params, plaintext_bytes[slot].as_ref());
                let scalar_poly = M::P::from_compact_bytes(local_params, scalar_bytes.as_ref());
                Arc::<[u8]>::from((plaintext * scalar_poly).to_compact_bytes())
            })
        });
        Self::new(
            params.clone(),
            vector_bytes,
            self.pubkey.large_scalar_mul(params, scalar),
            plaintext_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::BggPolyEncoding;
    use crate::{
        __PAIR, __TestState,
        bgg::{encoding::BggEncoding, public_key::BggPublicKey, sampler::BGGPublicKeySampler},
        circuit::evaluable::Evaluable,
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::hash::DCRTPolyHashSampler,
        utils::{block_size, create_random_poly},
    };
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;
    use std::path::Path;
    use tempfile::tempdir;

    fn random_row_vector(params: &DCRTPolyParams, columns: usize) -> DCRTPolyMatrix {
        DCRTPolyMatrix::from_poly_vec_row(
            params,
            (0..columns).map(|_| create_random_poly(params)).collect(),
        )
    }

    fn plaintext_bytes(plaintexts: &[DCRTPoly]) -> Vec<Vec<u8>> {
        plaintexts.iter().map(|plaintext| plaintext.to_compact_bytes()).collect()
    }

    fn vector_bytes(vectors: &[DCRTPolyMatrix]) -> Vec<Vec<u8>> {
        vectors.iter().map(|vector| vector.to_compact_bytes()).collect()
    }

    fn write_matrix_to_files(matrix: &DCRTPolyMatrix, dir: &Path, id: &str) {
        let entries_bytes = matrix
            .block_entries(0..matrix.row_size(), 0..matrix.col_size())
            .into_iter()
            .map(|row| row.into_iter().map(|poly| poly.to_compact_bytes()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let bytes = bincode::encode_to_vec(&entries_bytes, bincode::config::standard()).unwrap();
        let path = dir.join(format!(
            "{}_{}_{}.{}_{}.{}.matrix",
            id,
            block_size(),
            0,
            matrix.row_size(),
            0,
            matrix.col_size()
        ));
        std::fs::write(&path, bytes)
            .unwrap_or_else(|_| panic!("Failed to write matrix file {path:?}"));
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_poly_encoding_constructor_and_concat_vector() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let d = 3;
        let num_slots = 2;
        let columns = d * params.modulus_digits();
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &[true, true, true]);
        let pubkey = sampled_pub_keys[1].clone();

        let lhs_plaintexts =
            (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>();
        let rhs_plaintexts =
            (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>();
        let lhs_vectors =
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect::<Vec<_>>();
        let rhs_vectors =
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect::<Vec<_>>();
        let lhs = BggPolyEncoding::new(
            params.clone(),
            vector_bytes(&lhs_vectors),
            pubkey.clone(),
            Some(plaintext_bytes(&lhs_plaintexts)),
        );
        let rhs = BggPolyEncoding::new(
            params.clone(),
            vector_bytes(&rhs_vectors),
            pubkey,
            Some(plaintext_bytes(&rhs_plaintexts)),
        );

        let concatenated = lhs.concat_vector(std::slice::from_ref(&rhs));
        assert_eq!(concatenated.len(), num_slots);
        for (slot, concatenated_slot) in concatenated.iter().enumerate() {
            let lhs_slot = lhs.vector(slot);
            let rhs_slot = rhs.vector(slot);
            assert_eq!(concatenated_slot, &lhs_slot.concat_columns(&[&rhs_slot]));
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_poly_encoding_arithmetic() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let d = 3;
        let num_slots = 2;
        let columns = d * params.modulus_digits();
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &[true, true, true]);
        let lhs_pubkey = sampled_pub_keys[1].clone();
        let rhs_pubkey = sampled_pub_keys[2].clone();
        let lhs_plaintexts =
            (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>();
        let rhs_plaintexts =
            (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>();
        let lhs_vectors =
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect::<Vec<_>>();
        let rhs_vectors =
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect::<Vec<_>>();
        let lhs = BggPolyEncoding::new(
            params.clone(),
            vector_bytes(&lhs_vectors),
            lhs_pubkey.clone(),
            Some(plaintext_bytes(&lhs_plaintexts)),
        );
        let rhs = BggPolyEncoding::new(
            params.clone(),
            vector_bytes(&rhs_vectors),
            rhs_pubkey.clone(),
            Some(plaintext_bytes(&rhs_plaintexts)),
        );

        let expected_add = (0..num_slots)
            .map(|slot| {
                BggEncoding::new(lhs.vector(slot), lhs_pubkey.clone(), lhs.plaintext(slot)) +
                    &BggEncoding::new(rhs.vector(slot), rhs_pubkey.clone(), rhs.plaintext(slot))
            })
            .collect::<Vec<_>>();
        let addition = lhs.clone() + rhs.clone();
        assert_eq!(addition.pubkey, lhs_pubkey.clone() + rhs_pubkey.clone());
        for (slot, expected) in expected_add.iter().enumerate() {
            assert_eq!(addition.vector(slot), expected.vector);
            assert_eq!(addition.plaintext(slot), expected.plaintext.clone());
        }

        let expected_sub = (0..num_slots)
            .map(|slot| {
                BggEncoding::new(lhs.vector(slot), lhs_pubkey.clone(), lhs.plaintext(slot)) -
                    &BggEncoding::new(rhs.vector(slot), rhs_pubkey.clone(), rhs.plaintext(slot))
            })
            .collect::<Vec<_>>();
        let subtraction = lhs.clone() - rhs.clone();
        assert_eq!(subtraction.pubkey, lhs_pubkey.clone() - rhs_pubkey.clone());
        for (slot, expected) in expected_sub.iter().enumerate() {
            assert_eq!(subtraction.vector(slot), expected.vector);
            assert_eq!(subtraction.plaintext(slot), expected.plaintext.clone());
        }

        let expected_mul = (0..num_slots)
            .map(|slot| {
                BggEncoding::new(lhs.vector(slot), lhs_pubkey.clone(), lhs.plaintext(slot)) *
                    &BggEncoding::new(rhs.vector(slot), rhs_pubkey.clone(), rhs.plaintext(slot))
            })
            .collect::<Vec<_>>();
        let multiplication = lhs.clone() * rhs.clone();
        assert_eq!(multiplication.pubkey, lhs_pubkey * rhs_pubkey);
        for (slot, expected) in expected_mul.iter().enumerate() {
            assert_eq!(multiplication.vector(slot), expected.vector);
            assert_eq!(multiplication.plaintext(slot), expected.plaintext.clone());
        }
    }

    #[test]
    fn test_bgg_poly_encoding_new_rejects_slot_mismatch() {
        let params = DCRTPolyParams::default();
        let pubkey = BggPublicKey::new(DCRTPolyMatrix::identity(&params, 1, None), true);
        let panic = std::panic::catch_unwind(|| {
            let _ = BggPolyEncoding::new(
                params.clone(),
                vector_bytes(&[DCRTPolyMatrix::identity(&params, 1, None)]),
                pubkey,
                Some(plaintext_bytes(&[create_random_poly(&params), create_random_poly(&params)])),
            );
        })
        .expect_err("slot mismatch should panic");
        let panic_msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("panic payload should be a string");
        assert!(
            panic_msg.contains(
                "BggPolyEncoding::new requires plaintext_bytes.len() == vector_bytes.len()"
            )
        );
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_poly_encoding_evaluable_roundtrip_and_ops() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let d = 3;
        let num_slots = 2;
        let columns = d * params.modulus_digits();
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &[true, true, true]);
        let pubkey = sampled_pub_keys[1].clone();
        let encoding_plaintexts =
            (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>();
        let encoding_vectors =
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect::<Vec<_>>();
        let encoding = BggPolyEncoding::new(
            params.clone(),
            vector_bytes(&encoding_vectors),
            pubkey.clone(),
            Some(plaintext_bytes(&encoding_plaintexts)),
        );

        let compact = encoding.clone().to_compact();
        let restored = BggPolyEncoding::from_compact(&params, &compact);
        assert_eq!(restored, encoding);

        let rotated = encoding.rotate(&params, 1);
        assert_eq!(rotated.pubkey, pubkey.rotate(&params, 1));
        for slot in 0..num_slots {
            let expected =
                BggEncoding::new(encoding.vector(slot), pubkey.clone(), encoding.plaintext(slot))
                    .rotate(&params, 1);
            assert_eq!(rotated.vector(slot), expected.vector);
            assert_eq!(rotated.plaintext(slot), expected.plaintext);
        }

        let small_scaled = encoding.small_scalar_mul(&params, &[1, 2, 3]);
        assert_eq!(small_scaled.pubkey, pubkey.small_scalar_mul(&params, &[1, 2, 3]));
        for slot in 0..num_slots {
            let expected =
                BggEncoding::new(encoding.vector(slot), pubkey.clone(), encoding.plaintext(slot))
                    .small_scalar_mul(&params, &[1, 2, 3]);
            assert_eq!(small_scaled.vector(slot), expected.vector);
            assert_eq!(small_scaled.plaintext(slot), expected.plaintext);
        }

        let large_scalar = [BigUint::from(5u32), BigUint::from(8u32), BigUint::from(13u32)];
        let large_scaled = encoding.large_scalar_mul(&params, &large_scalar);
        assert_eq!(large_scaled.pubkey, pubkey.large_scalar_mul(&params, &large_scalar));
        for slot in 0..num_slots {
            let expected =
                BggEncoding::new(encoding.vector(slot), pubkey.clone(), encoding.plaintext(slot))
                    .large_scalar_mul(&params, &large_scalar);
            assert_eq!(large_scaled.vector(slot), expected.vector);
            assert_eq!(large_scaled.plaintext(slot), expected.plaintext);
        }
    }

    #[sequential_test::sequential]
    #[test]
    fn test_bgg_poly_encoding_read_from_files() {
        let key: [u8; 32] = rand::random();
        let tag: u64 = rand::random();
        let params = DCRTPolyParams::default();
        let d = 3;
        let num_slots = 2;
        let log_base_q = params.modulus_digits();
        let columns = d * log_base_q;
        let sampler = BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(key, d);
        let sampled_pub_keys = sampler.sample(&params, &tag.to_le_bytes(), &[true, true, true]);
        let pubkey = sampled_pub_keys[1].clone();
        let vectors =
            (0..num_slots).map(|_| random_row_vector(&params, columns)).collect::<Vec<_>>();
        let plaintexts = (0..num_slots).map(|_| create_random_poly(&params)).collect::<Vec<_>>();
        let dir = tempdir().unwrap();
        let dir_path = dir.path();

        for (slot, vector) in vectors.iter().enumerate() {
            write_matrix_to_files(vector, dir_path, &format!("sample_slot_{slot}_vector"));
        }
        write_matrix_to_files(&pubkey.matrix, dir_path, "sample_pubkey");
        for (slot, plaintext) in plaintexts.iter().enumerate() {
            plaintext.write_to_file(dir_path, &format!("sample_slot_{slot}_plaintext"));
        }

        let restored = BggPolyEncoding::<DCRTPolyMatrix>::read_from_files(
            &params, d, log_base_q, dir_path, "sample", true, num_slots,
        );

        assert_eq!(
            restored,
            BggPolyEncoding::new(
                params,
                vector_bytes(&vectors),
                pubkey,
                Some(plaintext_bytes(&plaintexts)),
            )
        );
    }
}
