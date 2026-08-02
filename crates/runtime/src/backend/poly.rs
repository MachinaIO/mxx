use super::{Backend, IndexRange, PreimageRequest, SampleRange};
use mxx_ir_core::{
    ParamEnv,
    node::{ConcatAxis, ConstantMatrix, HashVariant},
    types::ConcreteMatrixType,
};
use mxx_primitives::{
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
        hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
        uniform::DCRTPolyUniformSampler,
    },
};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use rayon::prelude::*;
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
    #[error(
        "declared gadget layout base={declared_base}, digits={declared_digits} does not match backend base={backend_base}, digits={backend_digits}"
    )]
    GadgetLayoutMismatch {
        declared_base: BigInt,
        declared_digits: usize,
        backend_base: BigInt,
        backend_digits: usize,
    },
}

pub struct PolyBackend<M, U, H, T>
where
    M: PolyMatrix,
{
    parameters: Vec<BTreeMap<RingKey, <M::P as Poly>::Params>>,
    active_placement: usize,
    preimage_batch_calls: usize,
    _marker: PhantomData<(M, U, H, T)>,
}

pub(crate) trait CrtRecomposeMatrix: PolyMatrix {
    fn crt_recompose_levels(
        levels: &[Self],
        plaintext_moduli: &[BigInt],
        reconstruction_coefficients: &[BigInt],
    ) -> Result<Self, PolyBackendError>;
}

pub(crate) fn crt_recompose_cpu<M: PolyMatrix>(
    levels: &[M],
    plaintext_moduli: &[BigInt],
    reconstruction_coefficients: &[BigInt],
) -> Result<M, PolyBackendError> {
    let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
    if levels.len() != plaintext_moduli.len() ||
        levels.len() != reconstruction_coefficients.len() ||
        levels.iter().any(|level| level.size() != first.size()) ||
        first.row_size() != 1
    {
        return Err(PolyBackendError::InvalidInteger);
    }
    let parameters = first.params();
    let modulus: Arc<BigUint> = parameters.modulus().into();
    let q = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
    let mut output = M::zero(parameters, 1, first.col_size());
    for ((level, plaintext_modulus), reconstruction_coefficient) in
        levels.iter().zip(plaintext_moduli).zip(reconstruction_coefficients)
    {
        let residue = ((reconstruction_coefficient % &q) + &q) % &q;
        let coefficient = M::P::from_biguint_to_constant(
            parameters,
            residue.to_biguint().ok_or(PolyBackendError::InvalidInteger)?,
        );
        let rounded = (0..first.col_size())
            .map(|column| {
                let coefficients = level
                    .entry(0, column)
                    .coeffs_biguints()
                    .into_iter()
                    .map(|value| {
                        let value = BigInt::from_biguint(Sign::Plus, value);
                        let rounded: BigInt =
                            ((plaintext_modulus * value + &q / 2) / &q) % plaintext_modulus;
                        rounded.to_biguint().ok_or(PolyBackendError::InvalidInteger)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(M::P::from_biguints(parameters, &coefficients))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        output.add_in_place(&(M::from_poly_vec_row(parameters, rounded) * coefficient));
    }
    Ok(output)
}

impl CrtRecomposeMatrix for DCRTPolyMatrix {
    fn crt_recompose_levels(
        levels: &[Self],
        plaintext_moduli: &[BigInt],
        reconstruction_coefficients: &[BigInt],
    ) -> Result<Self, PolyBackendError> {
        crt_recompose_cpu(levels, plaintext_moduli, reconstruction_coefficients)
    }
}

impl<M, U, H, T> Default for PolyBackend<M, U, H, T>
where
    M: PolyMatrix,
{
    fn default() -> Self {
        Self {
            parameters: vec![BTreeMap::new()],
            active_placement: 0,
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

    /// Constructs the production backend placements represented by the
    /// concrete parameter set.
    ///
    /// CPU parameters expose one logical placement. With the `gpu` feature,
    /// GPU parameters expose their configured device ids and each runtime
    /// placement receives an equivalent single-device parameter set. This
    /// keeps CRT limbs of one matrix together while allowing independent loop
    /// iterations to be scheduled across devices.
    pub fn new_for_execution(parameters: impl IntoIterator<Item = <M::P as Poly>::Params>) -> Self {
        Self::new_for_execution_on(parameters, &[])
    }

    /// Constructs production placements, preferring an application-selected
    /// device list over the device ids embedded in the parameter object.
    pub fn new_for_execution_on(
        parameters: impl IntoIterator<Item = <M::P as Poly>::Params>,
        requested_device_ids: &[i32],
    ) -> Self {
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        #[cfg(feature = "gpu")]
        {
            super::poly_gpu::new_for_execution_on(parameters, requested_device_ids)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = requested_device_ids;
            Self::new(parameters)
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn new_with_placements(placements: Vec<Vec<<M::P as Poly>::Params>>) -> Self {
        assert!(!placements.is_empty(), "a backend needs at least one placement");
        let mut backend = Self {
            parameters: (0..placements.len()).map(|_| BTreeMap::new()).collect(),
            active_placement: 0,
            preimage_batch_calls: 0,
            _marker: PhantomData,
        };
        for (placement, parameters) in placements.into_iter().enumerate() {
            for parameters in parameters {
                backend.register_at(placement, parameters);
            }
        }
        backend
    }

    fn register(&mut self, parameters: <M::P as Poly>::Params) {
        self.register_at(self.active_placement, parameters);
    }

    fn register_at(&mut self, placement: usize, parameters: <M::P as Poly>::Params) {
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let key = RingKey {
            modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
        };
        self.parameters[placement].insert(key, parameters);
    }

    /// Returns the number of batched preimage requests executed by this
    /// backend instance. This lightweight diagnostic distinguishes the
    /// bounded-wave batch path from scalar fallback execution.
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
        self.parameters[self.active_placement]
            .get(&key)
            .ok_or(PolyBackendError::MissingParameters(key))
    }

    pub(super) fn validate_regular_gadget_layout(
        parameters: &<M::P as Poly>::Params,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(), PolyBackendError> {
        let backend_base = BigInt::one() << parameters.base_bits() as usize;
        let backend_digits = parameters.modulus_digits();
        if gadget_base != &backend_base || digit_count != backend_digits {
            return Err(PolyBackendError::GadgetLayoutMismatch {
                declared_base: gadget_base.clone(),
                declared_digits: digit_count,
                backend_base,
                backend_digits,
            });
        }
        Ok(())
    }

    fn expected_gadget_layout(parameters: &<M::P as Poly>::Params, small: bool) -> (BigInt, usize) {
        let base = BigInt::one() << parameters.base_bits() as usize;
        let digits = if small {
            let (_, crt_bits, _) = parameters.to_crt();
            crt_bits.div_ceil(parameters.base_bits() as usize)
        } else {
            parameters.modulus_digits()
        };
        (base, digits)
    }

    fn parameters_for_matrix(
        &self,
        matrix: &M,
    ) -> Result<&<M::P as Poly>::Params, PolyBackendError> {
        let parameters = matrix.params();
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let key = RingKey {
            modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
        };
        self.parameters[self.active_placement]
            .get(&key)
            .ok_or(PolyBackendError::MissingParameters(key))
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
    M: CrtRecomposeMatrix + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    type Matrix = M;
    type Trapdoor = T::Trapdoor;
    type Error = PolyBackendError;

    fn placement_count(&self) -> usize {
        self.parameters.len()
    }

    fn active_placement(&self) -> usize {
        self.active_placement
    }

    fn set_active_placement(&mut self, placement: usize) -> bool {
        if placement >= self.parameters.len() {
            return false;
        }
        self.active_placement = placement;
        true
    }

    fn matrix_to_active_placement(&mut self, value: &M) -> Result<M, Self::Error> {
        let target = self.parameters_for_matrix(value)?;
        if value.params() == target {
            return Ok(value.clone());
        }
        let bytes = value.to_cpu_staging_bytes();
        Ok(M::from_cpu_staging_bytes(target, &bytes))
    }

    fn matrix_is_on_active_placement(&self, value: &M) -> bool {
        self.parameters_for_matrix(value).is_ok_and(|target| value.params() == target)
    }

    fn matrix_to_placements(&mut self, value: &M) -> Result<Vec<Option<M>>, Self::Error> {
        let source = value.params();
        let modulus: Arc<BigUint> = source.modulus().into();
        let key = RingKey {
            modulus: BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone()),
            ring_dimension: source.ring_dimension() as usize,
        };
        let targets = self
            .parameters
            .iter()
            .map(|parameters| {
                parameters.get(&key).ok_or_else(|| PolyBackendError::MissingParameters(key.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let staging =
            targets.iter().any(|target| source != *target).then(|| value.to_cpu_staging_bytes());
        Ok(targets
            .into_iter()
            .map(|target| {
                if source == target {
                    None
                } else {
                    Some(M::from_cpu_staging_bytes(
                        target,
                        staging.as_deref().expect("cross-placement staging was prepared"),
                    ))
                }
            })
            .collect())
    }

    fn trapdoor_to_active_placement(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &T::Trapdoor,
    ) -> Result<T::Trapdoor, Self::Error> {
        let parameters = self.parameters(ty)?;
        T::trapdoor_from_bytes(parameters, &T::trapdoor_to_bytes(value))
            .ok_or(PolyBackendError::TrapdoorDeserialization)
    }

    fn trapdoor_to_placements(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &T::Trapdoor,
        source_placement: usize,
    ) -> Result<Vec<T::Trapdoor>, Self::Error> {
        if self.parameters.len() == 1 {
            return Ok(vec![value.clone()]);
        }
        let bytes = T::trapdoor_to_bytes(value);
        self.parameters
            .iter()
            .enumerate()
            .map(|(placement, parameters)| {
                if placement == source_placement {
                    Ok(value.clone())
                } else {
                    let key =
                        RingKey { modulus: ty.modulus.clone(), ring_dimension: ty.ring_dimension };
                    T::trapdoor_from_bytes(
                        parameters
                            .get(&key)
                            .ok_or_else(|| PolyBackendError::MissingParameters(key.clone()))?,
                        &bytes,
                    )
                    .ok_or(PolyBackendError::TrapdoorDeserialization)
                }
            })
            .collect()
    }

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
            ConstantMatrix::Gadget { base, small } => {
                if !ty.columns.is_multiple_of(ty.rows) {
                    return Err(PolyBackendError::InvalidInteger);
                }
                let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                let digit_count = ty.columns / ty.rows;
                self.validate_gadget_layout(ty, &base, digit_count, *small)?;
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
            ConstantMatrix::Polynomial { coefficients } if ty.rows == 1 && ty.columns == 1 => {
                let modulus: Arc<BigUint> = parameters.modulus().into();
                let modulus = BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone());
                let coefficients = coefficients
                    .iter()
                    .map(|coefficient| {
                        coefficient
                            .evaluate(env)
                            .map_err(|_| PolyBackendError::InvalidInteger)?
                            .mod_floor(&modulus)
                            .to_biguint()
                            .ok_or(PolyBackendError::InvalidInteger)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                M::from_poly_vec(
                    parameters,
                    vec![vec![M::P::from_biguints(parameters, &coefficients)]],
                )
            }
            _ => return Err(PolyBackendError::InvalidConstantShape),
        })
    }

    fn add(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        Ok(left.clone() + right)
    }

    fn add_batch(&mut self, inputs: Vec<(M, M)>) -> Result<Vec<M>, Self::Error> {
        Ok(inputs.into_par_iter().map(|(left, right)| left + &right).collect())
    }

    fn sub(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        Ok(left.clone() - right)
    }

    fn sub_batch(&mut self, inputs: Vec<(M, M)>) -> Result<Vec<M>, Self::Error> {
        Ok(inputs.into_par_iter().map(|(left, right)| left - &right).collect())
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

    fn multiply_batch(&mut self, inputs: Vec<(M, M)>) -> Result<Vec<M>, Self::Error> {
        Ok(inputs
            .into_par_iter()
            .map(|(left, right)| {
                let left_size = left.size();
                let right_size = right.size();
                if left_size == (1, 1) {
                    right * left.entry(0, 0)
                } else if right_size == (1, 1) {
                    left * right.entry(0, 0)
                } else {
                    left * &right
                }
            })
            .collect())
    }

    fn negate(&mut self, value: &M) -> Result<M, Self::Error> {
        Ok(-value.clone())
    }

    fn negate_batch(&mut self, inputs: Vec<M>) -> Result<Vec<M>, Self::Error> {
        Ok(inputs.into_par_iter().map(|value| -value).collect())
    }

    fn scale_integer(&mut self, value: &M, scalar: &BigInt) -> Result<M, Self::Error> {
        let parameters = self.parameters_for_matrix(value)?;
        Ok(value.clone() * Self::ring_integer(parameters, scalar)?)
    }

    fn scale_integer_batch(&mut self, inputs: Vec<(M, BigInt)>) -> Result<Vec<M>, Self::Error> {
        let prepared = inputs
            .into_iter()
            .map(|(value, scalar)| {
                let parameters = self.parameters_for_matrix(&value)?;
                Ok((value, Self::ring_integer(parameters, &scalar)?))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        Ok(prepared.into_par_iter().map(|(value, scalar)| value * scalar).collect())
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

    fn concat(&mut self, inputs: &[&M], axis: ConcatAxis) -> Result<M, Self::Error> {
        let (first, rest) = inputs.split_first().ok_or(PolyBackendError::InvalidConstantShape)?;
        Ok(match axis {
            ConcatAxis::Rows => first.concat_rows(rest),
            ConcatAxis::Columns => first.concat_columns(rest),
            ConcatAxis::Diagonal => first.concat_diag(rest),
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
        Ok(if sigma == 0.0 {
            M::zero(parameters, ty.rows, ty.columns)
        } else {
            U::new().sample_uniform(parameters, ty.rows, ty.columns, DistType::GaussDist { sigma })
        })
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

    fn validate_gadget_layout(
        &self,
        ty: &ConcreteMatrixType,
        gadget_base: &BigInt,
        digit_count: usize,
        small: bool,
    ) -> Result<(), Self::Error> {
        let parameters = self.parameters(ty)?;
        let (backend_base, backend_digits) = Self::expected_gadget_layout(parameters, small);
        if gadget_base != &backend_base || digit_count != backend_digits {
            return Err(PolyBackendError::GadgetLayoutMismatch {
                declared_base: gadget_base.clone(),
                declared_digits: digit_count,
                backend_base,
                backend_digits,
            });
        }
        Ok(())
    }

    fn sample_trapdoor(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(M, T::Trapdoor), Self::Error> {
        let parameters = self.parameters(ty)?;
        Self::validate_regular_gadget_layout(parameters, gadget_base, digit_count)?;
        let sampler = T::new(parameters, sigma);
        let (trapdoor, public) = sampler.trapdoor(parameters, ty.rows);
        Ok((public, trapdoor))
    }

    fn sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        trapdoor: &T::Trapdoor,
        public: &M,
        target: &M,
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        Self::validate_regular_gadget_layout(parameters, gadget_base, digit_count)?;
        let sampler = T::new(parameters, sigma);
        Ok(sampler.preimage(parameters, trapdoor, public, target))
    }

    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<M, T::Trapdoor>>,
    ) -> Result<Vec<M>, Self::Error> {
        self.preimage_batch_calls += 1;
        #[cfg(not(feature = "gpu"))]
        {
            requests
                .into_iter()
                .map(|request| {
                    self.sample_preimage(
                        &request.matrix_type,
                        request.sigma,
                        &request.gadget_base,
                        request.digit_count,
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

    fn extract_coefficient(&mut self, value: &M, position: usize) -> Result<BigInt, Self::Error> {
        self.parameters_for_matrix(value)?;
        let residue = value
            .entry(0, 0)
            .coeffs_biguints()
            .get(position)
            .cloned()
            .ok_or(PolyBackendError::InvalidInteger)?;
        // `Int` values produced by coefficient extraction are used as family
        // and public-LUT indices. Preserve the canonical ring residue so a
        // valid coefficient above q/2 does not become a negative index.
        Ok(BigInt::from_biguint(Sign::Plus, residue))
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

    fn crt_recompose(
        &mut self,
        levels: &[M],
        plaintext_moduli: &[BigInt],
        reconstruction_coefficients: &[BigInt],
    ) -> Result<M, Self::Error> {
        M::crt_recompose_levels(levels, plaintext_moduli, reconstruction_coefficients)
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

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;

    #[test]
    fn coefficient_extraction_returns_a_canonical_index_above_half_modulus() {
        let parameters = DCRTPolyParams::new(2, 1, 10, 5);
        let modulus = parameters.modulus();
        let residue = modulus.as_ref() - BigUint::from(1u8);
        let value = DCRTPolyMatrix::from_poly_vec_row(
            &parameters,
            vec![DCRTPoly::from_biguint_to_constant(&parameters, residue.clone())],
        );
        let mut backend = cpu_backend([parameters]);

        assert_eq!(
            backend.extract_coefficient(&value, 0).expect("extract coefficient"),
            BigInt::from_biguint(Sign::Plus, residue)
        );
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use crate::backend::poly_gpu::{GpuDcrtBackend, gpu_backend};
}
