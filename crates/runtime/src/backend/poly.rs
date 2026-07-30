use super::{Backend, PreimageRequest};
use mxx_ir_core::{
    ParamEnv,
    node::{ConcatAxis, ConstantMatrix, HashVariant, IndexRange, SampleRange},
    types::ConcreteMatrixType,
};
use mxx_primitives::{
    element::PolyElem,
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    modulus::{ModulusRaiseError, modulus_raise},
    poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
        hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
        uniform::DCRTPolyUniformSampler,
    },
};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{One, ToPrimitive, Zero};
use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RingKey {
    pub modulus: BigInt,
    pub ring_dimension: usize,
}

#[derive(Debug, Error)]
pub enum PolyBackendError {
    #[error("no concrete polynomial parameters registered for {0:?}")]
    MissingParameters(RingKey),
    #[error("uniform range [{minimum}, {maximum}] is not supported by existing samplers")]
    UnsupportedUniformRange { minimum: BigInt, maximum: BigInt },
    #[error("matrix shape is invalid for this constant")]
    InvalidConstantShape,
    #[error("integer value cannot be represented in the target ring")]
    InvalidInteger,
    #[error("trapdoor deserialization failed")]
    TrapdoorDeserialization,
    #[error("matrix is empty")]
    EmptyMatrix,
    #[error(transparent)]
    ModulusRaise(#[from] ModulusRaiseError),
}

pub struct PolyBackend<M, U, H, T>
where
    M: PolyMatrix,
{
    parameters: BTreeMap<RingKey, <M::P as Poly>::Params>,
    #[cfg(test)]
    preimage_batch_calls: usize,
    _marker: PhantomData<(M, U, H, T)>,
}

impl<M, U, H, T> Default for PolyBackend<M, U, H, T>
where
    M: PolyMatrix,
{
    fn default() -> Self {
        Self {
            parameters: BTreeMap::new(),
            #[cfg(test)]
            preimage_batch_calls: 0,
            _marker: PhantomData,
        }
    }
}

impl<M, U, H, T> PolyBackend<M, U, H, T>
where
    M: PolyMatrix,
{
    pub fn new(parameters: impl IntoIterator<Item = <M::P as Poly>::Params>) -> Self {
        let mut backend = Self::default();
        for parameters in parameters {
            backend.register(parameters);
        }
        backend
    }

    pub fn register(&mut self, parameters: <M::P as Poly>::Params) {
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let key = RingKey {
            modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
        };
        self.parameters.insert(key, parameters);
    }

    #[cfg(test)]
    pub fn preimage_batch_calls(&self) -> usize {
        self.preimage_batch_calls
    }

    pub(super) fn parameters(
        &self,
        matrix_type: &ConcreteMatrixType,
    ) -> Result<&<M::P as Poly>::Params, PolyBackendError> {
        let key = RingKey {
            modulus: matrix_type.modulus.clone(),
            ring_dimension: matrix_type.ring_dimension,
        };
        self.parameters.get(&key).ok_or(PolyBackendError::MissingParameters(key))
    }

    fn parameters_for_matrix(
        &self,
        matrix: &M,
    ) -> Result<&<M::P as Poly>::Params, PolyBackendError> {
        let (rows, columns) = matrix.size();
        if rows == 0 || columns == 0 {
            return Err(PolyBackendError::EmptyMatrix);
        }
        let polynomial = matrix.entry(0, 0);
        let coefficient =
            polynomial.coeffs().into_iter().next().ok_or(PolyBackendError::EmptyMatrix)?;
        let modulus: Arc<BigUint> = coefficient.modulus().clone().into();
        let ring_dimension = polynomial.coeffs().len();
        let key = RingKey {
            modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
            ring_dimension,
        };
        self.parameters.get(&key).ok_or(PolyBackendError::MissingParameters(key))
    }

    fn ring_integer(
        parameters: &<M::P as Poly>::Params,
        value: &BigInt,
    ) -> Result<M::P, PolyBackendError> {
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let modulus_int = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
        let residue = ((value % &modulus_int) + &modulus_int) % &modulus_int;
        let residue = residue.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
        Ok(M::P::from_biguint_to_constant(parameters, residue))
    }
}

impl<M, U, H, T> Backend for PolyBackend<M, U, H, T>
where
    M: PolyMatrix + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    type Matrix = M;
    type Trapdoor = T::Trapdoor;
    type Error = PolyBackendError;

    fn constant_matrix(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &ConstantMatrix,
        env: &ParamEnv,
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        Ok(match value {
            ConstantMatrix::Zero => M::zero(parameters, ty.rows, ty.columns),
            ConstantMatrix::Identity if ty.rows == ty.columns => {
                M::identity(parameters, ty.rows, None)
            }
            ConstantMatrix::UnitRow { index } if ty.rows == 1 => {
                let index = index
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(PolyBackendError::InvalidInteger)?;
                M::unit_row_vector(parameters, ty.columns, index)
            }
            ConstantMatrix::UnitColumn { index } if ty.columns == 1 => {
                let index = index
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(PolyBackendError::InvalidInteger)?;
                M::unit_column_vector(parameters, ty.rows, index)
            }
            ConstantMatrix::Gadget { small, .. } => {
                if *small {
                    M::small_gadget_matrix(parameters, ty.rows)
                } else {
                    M::gadget_matrix(parameters, ty.rows)
                }
            }
            ConstantMatrix::PowerOfBase { base, exponent } if ty.rows == 1 && ty.columns == 1 => {
                let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                let exponent = exponent
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_u32())
                    .ok_or(PolyBackendError::InvalidInteger)?;
                let value = base.pow(exponent);
                M::from_poly_vec(parameters, vec![vec![Self::ring_integer(parameters, &value)?]])
            }
            ConstantMatrix::Rotation { exponent } if ty.rows == 1 && ty.columns == 1 => {
                let exponent = exponent
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(PolyBackendError::InvalidInteger)?;
                M::from_poly_vec(
                    parameters,
                    vec![vec![M::P::const_rotate_poly(parameters, exponent)]],
                )
            }
            _ => return Err(PolyBackendError::InvalidConstantShape),
        })
    }

    fn add(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        Ok(left.clone() + right)
    }

    fn sub(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        Ok(left.clone() - right)
    }

    fn multiply(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        let left_size = left.size();
        let right_size = right.size();
        Ok(if left_size == (1, 1) {
            right.clone() * left.entry(0, 0)
        } else if right_size == (1, 1) {
            left.clone() * right.entry(0, 0)
        } else {
            left.clone() * right
        })
    }

    fn negate(&mut self, value: &M) -> Result<M, Self::Error> {
        Ok(-value.clone())
    }

    fn scale_integer(&mut self, value: &M, scalar: &BigInt) -> Result<M, Self::Error> {
        let parameters = self.parameters_for_matrix(value)?;
        Ok(value.clone() * Self::ring_integer(parameters, scalar)?)
    }

    fn transpose(&mut self, value: &M) -> Result<M, Self::Error> {
        Ok(value.transpose())
    }

    fn slice(
        &mut self,
        value: &M,
        rows: Option<&IndexRange>,
        columns: Option<&IndexRange>,
    ) -> Result<M, Self::Error> {
        let (row_count, column_count) = value.size();
        let rows = rows.cloned().unwrap_or(IndexRange { start: 0, end: row_count });
        let columns = columns.cloned().unwrap_or(IndexRange { start: 0, end: column_count });
        Ok(value.slice(rows.start, rows.end, columns.start, columns.end))
    }

    fn tensor(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        Ok(left.tensor(right))
    }

    fn concat(&mut self, inputs: &[M], axis: ConcatAxis) -> Result<M, Self::Error> {
        let (first, rest) = inputs.split_first().ok_or(PolyBackendError::InvalidConstantShape)?;
        let references = rest.iter().collect::<Vec<_>>();
        Ok(match axis {
            ConcatAxis::Rows => first.concat_rows(&references),
            ConcatAxis::Columns => first.concat_columns(&references),
            ConcatAxis::Diagonal => first.concat_diag(&references),
        })
    }

    fn reshape(&mut self, value: &M, rows: usize, columns: usize) -> Result<M, Self::Error> {
        let parameters = self.parameters_for_matrix(value)?;
        let (old_rows, old_columns) = value.size();
        if old_rows.saturating_mul(old_columns) != rows.saturating_mul(columns) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let entries = (0..old_rows)
            .flat_map(|row| (0..old_columns).map(move |column| value.entry(row, column)))
            .collect::<Vec<_>>();
        let entries = entries.chunks(columns).map(|row| row.to_vec()).collect::<Vec<_>>();
        Ok(M::from_poly_vec(parameters, entries))
    }

    fn sample_uniform(
        &mut self,
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let maximum = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()) - 1;
        let distribution = if range.minimum == BigInt::from(-1) && range.maximum == BigInt::from(1)
        {
            DistType::TernaryDist
        } else if range.minimum.is_zero() && range.maximum == BigInt::one() {
            DistType::BitDist
        } else if range.minimum.is_zero() && range.maximum == maximum {
            DistType::FinRingDist
        } else {
            return Err(PolyBackendError::UnsupportedUniformRange {
                minimum: range.minimum.clone(),
                maximum: range.maximum.clone(),
            });
        };
        Ok(U::new().sample_uniform(parameters, ty.rows, ty.columns, distribution))
    }

    fn sample_gaussian(&mut self, ty: &ConcreteMatrixType, sigma: f64) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        Ok(U::new().sample_uniform(parameters, ty.rows, ty.columns, DistType::GaussDist { sigma }))
    }

    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        variant: HashVariant,
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        let sampler = H::new();
        Ok(match variant {
            HashVariant::Plain => sampler.sample_hash(
                parameters,
                key,
                tag,
                ty.rows,
                ty.columns,
                DistType::FinRingDist,
            ),
            HashVariant::Decomposed => {
                let digits = parameters.modulus_digits();
                sampler.sample_hash_decomposed(
                    parameters,
                    key,
                    tag,
                    ty.rows / digits,
                    ty.columns,
                    DistType::FinRingDist,
                )
            }
            HashVariant::SmallDecomposed => {
                let (_, crt_bits, _) = parameters.to_crt();
                let digits = crt_bits.div_ceil(parameters.base_bits() as usize);
                sampler.sample_hash_small_decomposed(
                    parameters,
                    key,
                    tag,
                    ty.rows / digits,
                    ty.columns,
                    DistType::FinRingDist,
                )
            }
        })
    }

    fn sample_trapdoor(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
    ) -> Result<(M, T::Trapdoor), Self::Error> {
        let parameters = self.parameters(ty)?;
        let sampler = T::new(parameters, sigma);
        let (trapdoor, public) = sampler.trapdoor(parameters, ty.rows);
        Ok((public, trapdoor))
    }

    fn sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        trapdoor: &T::Trapdoor,
        public: &M,
        target: &M,
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        let sampler = T::new(parameters, sigma);
        Ok(sampler.preimage(parameters, trapdoor, public, target))
    }

    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<M, T::Trapdoor>>,
    ) -> Result<Vec<M>, Self::Error> {
        #[cfg(test)]
        {
            self.preimage_batch_calls += 1;
        }
        #[cfg(not(feature = "gpu"))]
        {
            requests
                .into_iter()
                .map(|request| {
                    self.sample_preimage(
                        &request.matrix_type,
                        request.sigma,
                        &request.trapdoor,
                        &request.public,
                        &request.target,
                    )
                })
                .collect()
        }
        #[cfg(feature = "gpu")]
        {
            super::poly_gpu::sample_preimage_batch(self, requests)
        }
    }

    fn gadget_decompose(&mut self, value: &M, small: bool) -> Result<M, Self::Error> {
        Ok(if small { value.small_decompose() } else { value.decompose() })
    }

    fn modulus_down(&mut self, value: &M, target_modulus: &BigInt) -> Result<M, Self::Error> {
        let source = self.parameters_for_matrix(value)?;
        let target_key = RingKey {
            modulus: target_modulus.clone(),
            ring_dimension: source.ring_dimension() as usize,
        };
        let target = self
            .parameters
            .get(&target_key)
            .ok_or(PolyBackendError::MissingParameters(target_key))?;
        Ok(value.modulus_switch(&target.modulus()))
    }

    fn modulus_up(
        &mut self,
        value: &M,
        target_type: &ConcreteMatrixType,
    ) -> Result<M, Self::Error> {
        let source = self.parameters_for_matrix(value)?;
        let target = self.parameters(target_type)?;
        Ok(modulus_raise(value, source, target)?)
    }

    fn extract_coefficient(&mut self, value: &M, position: usize) -> Result<BigInt, Self::Error> {
        let parameters = self.parameters_for_matrix(value)?;
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let residue = value
            .entry(0, 0)
            .coeffs_biguints()
            .get(position)
            .cloned()
            .ok_or(PolyBackendError::InvalidInteger)?;
        if &residue * BigUint::from(2u8) > *modulus {
            Ok(-BigInt::from_biguint(Sign::Plus, modulus.as_ref() - residue))
        } else {
            Ok(BigInt::from_biguint(Sign::Plus, residue))
        }
    }

    fn threshold_decode(
        &mut self,
        value: &M,
        plaintext_modulus: &BigInt,
        length: usize,
    ) -> Result<Vec<BigInt>, Self::Error> {
        let parameters = self.parameters_for_matrix(value)?;
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let q = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
        let coefficients = value.entry(0, 0).coeffs_biguints();
        Ok(coefficients
            .into_iter()
            .take(length)
            .map(|coefficient| {
                let coefficient = BigInt::from_biguint(Sign::Plus, coefficient);
                ((plaintext_modulus * coefficient + &q / 2) / &q) % plaintext_modulus
            })
            .collect())
    }

    fn matrix_to_bytes(&self, value: &M) -> Vec<u8> {
        value.to_compact_bytes()
    }

    fn matrix_from_bytes(&self, ty: &ConcreteMatrixType, bytes: &[u8]) -> Result<M, Self::Error> {
        Ok(M::from_compact_bytes(self.parameters(ty)?, bytes))
    }

    fn trapdoor_to_bytes(&self, value: &T::Trapdoor) -> Vec<u8> {
        T::trapdoor_to_bytes(value)
    }

    fn trapdoor_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<T::Trapdoor, Self::Error> {
        T::trapdoor_from_bytes(self.parameters(ty)?, bytes)
            .ok_or(PolyBackendError::TrapdoorDeserialization)
    }
}

pub type CpuDcrtBackend = PolyBackend<
    DCRTPolyMatrix,
    DCRTPolyUniformSampler,
    DCRTPolyHashSampler<keccak_asm::Keccak256>,
    DCRTPolyTrapdoorSampler,
>;

pub fn cpu_backend(parameters: impl IntoIterator<Item = DCRTPolyParams>) -> CpuDcrtBackend {
    CpuDcrtBackend::new(parameters)
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use mxx_primitives::{
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::dcrt::gpu::GpuDCRTPolyParams,
        sampler::{
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
    };

    pub type GpuDcrtBackend = PolyBackend<
        GpuDCRTPolyMatrix,
        GpuDCRTPolyUniformSampler,
        GpuDCRTPolyHashSampler<keccak_asm::Keccak256>,
        GpuDCRTPolyTrapdoorSampler,
    >;

    pub fn gpu_backend(parameters: impl IntoIterator<Item = GpuDCRTPolyParams>) -> GpuDcrtBackend {
        GpuDcrtBackend::new(parameters)
    }
}
