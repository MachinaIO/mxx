use super::{Backend, IndexRange, MatrixMulAccumulateRequest, PreimageRequest, SampleRange};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix},
    types::ConcreteMatrixType,
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixSmallRhs, SmallMatrixError, SmallPolyMatrix,
        dcrt_poly::DCRTPolyMatrix,
    },
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

const SMALL_MATRIX_MAGIC: &[u8; 4] = b"SMR1";

fn small_matrix_semantic_tag(kind: SmallMatrixSemanticKind) -> u8 {
    match kind {
        SmallMatrixSemanticKind::Generic => 0,
        SmallMatrixSemanticKind::Preimage => 1,
    }
}

fn take_small_matrix_bytes<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    length: usize,
) -> Result<&'a [u8], PolyBackendError> {
    let end = offset
        .checked_add(length)
        .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("header length overflows"))?;
    let value = bytes
        .get(*offset..end)
        .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("header is truncated"))?;
    *offset = end;
    Ok(value)
}

fn read_small_matrix_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, PolyBackendError> {
    let raw: [u8; 4] = take_small_matrix_bytes(bytes, offset, 4)?
        .try_into()
        .expect("four-byte slice has fixed width");
    Ok(u32::from_le_bytes(raw))
}

fn read_small_matrix_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, PolyBackendError> {
    let raw: [u8; 8] = take_small_matrix_bytes(bytes, offset, 8)?
        .try_into()
        .expect("eight-byte slice has fixed width");
    Ok(u64::from_le_bytes(raw))
}

fn bounded_schema_parts(
    schema: &ConcreteBoundedMatrixSchema,
) -> Result<(BigUint, usize, usize), PolyBackendError> {
    let bound = schema
        .max_coefficient_bound
        .to_biguint()
        .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("bound is negative"))?;
    let magnitude_bytes = usize::try_from(bound.bits().div_ceil(8))
        .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("bound width overflows"))?
        .max(1);
    let coefficient_count = schema
        .matrix
        .rows
        .checked_mul(schema.matrix.columns)
        .and_then(|count| count.checked_mul(schema.matrix.ring_dimension))
        .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("coefficient count overflows"))?;
    Ok((bound, magnitude_bytes, coefficient_count))
}

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
    #[error(transparent)]
    SmallMatrix(#[from] SmallMatrixError),
    #[error("invalid small-matrix artifact: {0}")]
    InvalidSmallMatrixArtifact(&'static str),
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

fn validate_regular_gadget_layout_for_params<P: PolyParams>(
    parameters: &P,
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

fn sample_preimage_with_parameters<M, T>(
    parameters: &<M::P as Poly>::Params,
    request: PreimageRequest<M, T::Trapdoor>,
) -> Result<M::SmallMatrix, PolyBackendError>
where
    M: PolyMatrixSmallRhs,
    T: PolyTrapdoorSampler<M = M>,
{
    validate_regular_gadget_layout_for_params(
        parameters,
        &request.gadget_base,
        request.digit_count,
    )?;
    let max_coefficient_bound =
        request.max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
    Ok(T::new(parameters, request.sigma).preimage(
        parameters,
        request.trapdoor.as_ref(),
        request.public.as_ref(),
        request.target.as_ref(),
        max_coefficient_bound,
    )?)
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

    fn parameters_at(
        &self,
        placement: usize,
        matrix_type: &ConcreteMatrixType,
    ) -> Result<&<M::P as Poly>::Params, PolyBackendError> {
        let key = RingKey {
            modulus: matrix_type.modulus.clone(),
            ring_dimension: matrix_type.ring_dimension,
        };
        self.parameters
            .get(placement)
            .and_then(|parameters| parameters.get(&key))
            .ok_or(PolyBackendError::MissingParameters(key))
    }

    fn validate_regular_gadget_layout(
        parameters: &<M::P as Poly>::Params,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(), PolyBackendError> {
        validate_regular_gadget_layout_for_params(parameters, gadget_base, digit_count)
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

    fn parameters_for_small_matrix(
        &self,
        matrix: &M::SmallMatrix,
    ) -> Result<&<M::P as Poly>::Params, PolyBackendError>
    where
        M: PolyMatrixSmallRhs,
    {
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
    M: CrtRecomposeMatrix + PolyMatrixSmallRhs + 'static,
    U: PolyUniformSampler<M = M>,
    H: PolyHashSampler<[u8; 32], M = M>,
    T: PolyTrapdoorSampler<M = M>,
    T::Trapdoor: Clone + std::fmt::Debug,
{
    type Matrix = M;
    type SmallMatrix = M::SmallMatrix;
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
        if let Some(copied) = value.copy_to_params_direct(target) {
            return Ok(copied);
        }
        let bytes = value.to_cpu_staging_bytes();
        Ok(M::from_cpu_staging_bytes(target, &bytes))
    }

    fn matrix_is_on_active_placement(&self, value: &M) -> bool {
        self.parameters_for_matrix(value).is_ok_and(|target| value.params() == target)
    }

    fn small_matrix_to_active_placement(
        &mut self,
        value: &M::SmallMatrix,
    ) -> Result<M::SmallMatrix, Self::Error> {
        let target = self.parameters_for_small_matrix(value)?;
        if value.params() == target {
            return Ok(value.clone());
        }
        let payload = value.to_canonical_coefficients()?;
        Ok(M::SmallMatrix::from_canonical_coefficients(
            target,
            value.rows(),
            value.columns(),
            value.max_coefficient_bound().clone(),
            &payload,
        )?)
    }

    fn small_matrix_is_on_active_placement(&self, value: &M::SmallMatrix) -> bool {
        self.parameters_for_small_matrix(value).is_ok_and(|target| value.params() == target)
    }

    fn fence_released_memory(&mut self) -> Result<(), Self::Error> {
        for placement in &self.parameters {
            for parameters in placement.values() {
                parameters.fence_released_memory();
            }
        }
        Ok(())
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
        let mut outputs = (0..targets.len()).map(|_| None).collect::<Vec<_>>();
        let mut staged_indices = Vec::new();
        for (index, target) in targets.iter().enumerate() {
            if source == *target {
                continue;
            }
            if let Some(copied) = value.copy_to_params_direct(target) {
                outputs[index] = Some(copied);
            } else {
                staged_indices.push(index);
            }
        }
        let staged_targets = staged_indices.iter().map(|index| targets[*index]).collect::<Vec<_>>();
        for (index, copied) in
            staged_indices.into_iter().zip(value.copy_to_params_fanout(&staged_targets))
        {
            outputs[index] = Some(copied);
        }
        Ok(outputs)
    }

    fn small_matrix_to_placements(
        &mut self,
        value: &M::SmallMatrix,
    ) -> Result<Vec<Option<M::SmallMatrix>>, Self::Error> {
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
        let mut payload = None;
        targets
            .into_iter()
            .map(|target| {
                if source == target {
                    return Ok(None);
                }
                let payload = match &payload {
                    Some(payload) => payload,
                    None => payload.insert(value.to_canonical_coefficients()?),
                };
                Ok(Some(M::SmallMatrix::from_canonical_coefficients(
                    target,
                    value.rows(),
                    value.columns(),
                    value.max_coefficient_bound().clone(),
                    payload,
                )?))
            })
            .collect()
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
        Ok(left.add_out_of_place(right))
    }

    fn add_batch(&mut self, inputs: Vec<(Arc<M>, Arc<M>)>) -> Result<Vec<M>, Self::Error> {
        Ok(M::add_batch_out_of_place(inputs))
    }

    fn sub(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        Ok(left.sub_out_of_place(right))
    }

    fn sub_batch(&mut self, inputs: Vec<(Arc<M>, Arc<M>)>) -> Result<Vec<M>, Self::Error> {
        Ok(M::sub_batch_out_of_place(inputs))
    }

    fn multiply(&mut self, left: &M, right: &M) -> Result<M, Self::Error> {
        let left_size = left.size();
        let right_size = right.size();
        Ok(if left_size == (1, 1) {
            right.multiply_poly_out_of_place(&left.entry(0, 0))
        } else if right_size == (1, 1) {
            left.multiply_poly_out_of_place(&right.entry(0, 0))
        } else {
            left.multiply_out_of_place(right)
        })
    }

    fn multiply_batch(&mut self, inputs: Vec<(Arc<M>, Arc<M>)>) -> Result<Vec<M>, Self::Error> {
        Ok(M::multiply_batch_out_of_place(inputs))
    }

    fn matrix_mul_accumulate_batch(
        &mut self,
        requests: Vec<MatrixMulAccumulateRequest<M>>,
    ) -> Result<Vec<M>, Self::Error> {
        let requests = requests
            .into_iter()
            .map(|request| {
                let parameters = request
                    .products
                    .first()
                    .expect("validated multi-row GEMM has a product")
                    .1
                    .params()
                    .clone();
                let products = request
                    .products
                    .into_iter()
                    .map(|(coefficient, left, right)| {
                        let coefficient = if coefficient.is_one() {
                            None
                        } else {
                            Some(Self::ring_integer(&parameters, &coefficient)?)
                        };
                        Ok((coefficient, left, right))
                    })
                    .collect::<Result<Vec<_>, PolyBackendError>>()?;
                Ok((products, request.bias))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        Ok(M::multiply_accumulate_batch_out_of_place(requests))
    }

    fn negate(&mut self, value: &M) -> Result<M, Self::Error> {
        Ok(value.negate_out_of_place())
    }

    fn negate_batch(&mut self, inputs: Vec<Arc<M>>) -> Result<Vec<M>, Self::Error> {
        Ok(M::negate_batch_out_of_place(inputs))
    }

    fn scale_integer(&mut self, value: &M, scalar: &BigInt) -> Result<M, Self::Error> {
        let parameters = self.parameters_for_matrix(value)?;
        Ok(value.multiply_poly_out_of_place(&Self::ring_integer(parameters, scalar)?))
    }

    fn scale_integer_batch(
        &mut self,
        inputs: Vec<(Arc<M>, BigInt)>,
    ) -> Result<Vec<M>, Self::Error> {
        let prepared = inputs
            .into_iter()
            .map(|(value, scalar)| {
                let parameters = self.parameters_for_matrix(&value)?;
                Ok((value, Self::ring_integer(parameters, &scalar)?))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        Ok(M::multiply_polys_batch_out_of_place(prepared))
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

    fn sample_gaussian(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: &BigInt,
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        let max_coefficient_bound =
            max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
        Ok(if sigma == 0.0 {
            M::zero(parameters, ty.rows, ty.columns)
        } else {
            U::new().sample_uniform(
                parameters,
                ty.rows,
                ty.columns,
                DistType::GaussDist { sigma, max_coefficient_bound: Some(max_coefficient_bound) },
            )
        })
    }

    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        Ok(H::new().sample_hash(parameters, key, tag, ty.rows, ty.columns, DistType::FinRingDist))
    }

    fn sample_hash_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<M::SmallMatrix, Self::Error> {
        self.validate_gadget_layout(ty, gadget_base, digit_count, false)?;
        if digit_count == 0 || !ty.rows.is_multiple_of(digit_count) {
            return Err(PolyBackendError::InvalidInteger);
        }
        let parameters = self.parameters(ty)?;
        let source = H::new().sample_hash(
            parameters,
            key,
            tag,
            ty.rows / digit_count,
            ty.columns,
            DistType::FinRingDist,
        );
        Ok(source.gadget_decompose(false)?)
    }

    fn sample_hash_small_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<M::SmallMatrix, Self::Error> {
        self.validate_gadget_layout(ty, gadget_base, digit_count, true)?;
        if digit_count == 0 || !ty.rows.is_multiple_of(digit_count) {
            return Err(PolyBackendError::InvalidInteger);
        }
        let parameters = self.parameters(ty)?;
        let source = H::new().sample_hash(
            parameters,
            key,
            tag,
            ty.rows / digit_count,
            ty.columns,
            DistType::FinRingDist,
        );
        Ok(source.gadget_decompose(true)?)
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
        max_coefficient_bound: &BigInt,
        trapdoor: &T::Trapdoor,
        public: &M,
        target: &M,
    ) -> Result<M::SmallMatrix, Self::Error> {
        let parameters = self.parameters(ty)?;
        Self::validate_regular_gadget_layout(parameters, gadget_base, digit_count)?;
        let max_coefficient_bound =
            max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
        let sampler = T::new(parameters, sigma);
        Ok(sampler.preimage(parameters, trapdoor, public, target, max_coefficient_bound)?)
    }

    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<M, T::Trapdoor>>,
    ) -> Result<Vec<M::SmallMatrix>, Self::Error> {
        self.preimage_batch_calls += 1;
        let Some(first) = requests.first() else {
            return Ok(Vec::new());
        };
        let parameters = self.parameters(&first.matrix_type)?;
        requests
            .into_iter()
            .map(|request| sample_preimage_with_parameters::<M, T>(parameters, request))
            .collect()
    }

    fn sample_preimage_batches_by_placement(
        &mut self,
        batches: Vec<(usize, Vec<PreimageRequest<M, T::Trapdoor>>)>,
    ) -> Result<Vec<(usize, Vec<M::SmallMatrix>)>, Self::Error> {
        self.preimage_batch_calls += batches.len();
        let prepared = batches
            .into_iter()
            .map(|(placement, requests)| {
                let first = requests.first().ok_or(PolyBackendError::InvalidInteger)?;
                Ok((placement, self.parameters_at(placement, &first.matrix_type)?, requests))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        prepared
            .into_par_iter()
            .map(|(placement, parameters, requests)| {
                requests
                    .into_iter()
                    .map(|request| sample_preimage_with_parameters::<M, T>(parameters, request))
                    .collect::<Result<Vec<_>, _>>()
                    .map(|outputs| (placement, outputs))
            })
            .collect()
    }

    fn gadget_decompose(&mut self, value: &M, small: bool) -> Result<M::SmallMatrix, Self::Error> {
        Ok(value.clone().gadget_decompose(small)?)
    }

    fn multiply_small_rhs(&mut self, lhs: &M, rhs: &M::SmallMatrix) -> Result<M, Self::Error> {
        Ok(lhs.multiply_small_rhs(rhs.clone())?)
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

    fn pack_polynomial_coefficients(
        &mut self,
        ty: &ConcreteMatrixType,
        bits: &[bool],
        coefficient_bits: usize,
    ) -> Result<M, Self::Error> {
        if !ty.is_scalar() ||
            coefficient_bits == 0 ||
            bits.len() != ty.ring_dimension.saturating_mul(coefficient_bits)
        {
            return Err(PolyBackendError::InvalidInteger);
        }
        let parameters = self.parameters(ty)?;
        let modulus: Arc<BigUint> = parameters.modulus().into();
        let coefficients = bits
            .chunks_exact(coefficient_bits)
            .map(|coefficient_bits| {
                let mut coefficient = BigUint::zero();
                for (position, bit) in coefficient_bits.iter().copied().enumerate() {
                    if bit {
                        coefficient |= BigUint::one() << position;
                    }
                }
                (coefficient < *modulus)
                    .then_some(coefficient)
                    .ok_or(PolyBackendError::InvalidInteger)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(M::from_poly_vec_row(parameters, vec![M::P::from_biguints(parameters, &coefficients)]))
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

    fn matrices_to_bytes(&self, values: &[&M]) -> Vec<Vec<u8>> {
        #[cfg(feature = "gpu")]
        {
            M::compact_bytes_batch(values)
        }
        #[cfg(not(feature = "gpu"))]
        {
            values.iter().map(|value| value.to_compact_bytes()).collect()
        }
    }

    fn matrix_from_bytes(&self, ty: &ConcreteMatrixType, bytes: &[u8]) -> Result<M, Self::Error> {
        Ok(M::from_compact_bytes(self.parameters(ty)?, bytes))
    }

    fn small_matrix_to_bytes(
        &self,
        value: &M::SmallMatrix,
        expected_schema: &ConcreteBoundedMatrixSchema,
        semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Vec<u8>, Self::Error> {
        let parameters = self.parameters(&expected_schema.matrix)?;
        let (bound, magnitude_bytes, coefficient_count) = bounded_schema_parts(expected_schema)?;
        value.validate_metadata(
            parameters,
            expected_schema.matrix.rows,
            expected_schema.matrix.columns,
            &bound,
        )?;
        let payload = value.to_canonical_coefficients()?;
        let encoded_width = 1usize
            .checked_add(magnitude_bytes)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("coefficient width overflows"))?;
        let expected_payload_length = coefficient_count
            .checked_mul(encoded_width)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("payload length overflows"))?;
        if payload.len() != expected_payload_length {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                "owner returned a payload with the wrong length",
            ));
        }
        let bound_bytes = {
            let bytes = bound.to_bytes_le();
            if bytes.is_empty() { vec![0] } else { bytes }
        };
        if bound_bytes.len() != magnitude_bytes {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                "bound width is not canonical",
            ));
        }
        let rows = u64::try_from(expected_schema.matrix.rows)
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("row count overflows"))?;
        let columns = u64::try_from(expected_schema.matrix.columns)
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("column count overflows"))?;
        let ring_dimension =
            u64::try_from(expected_schema.matrix.ring_dimension).map_err(|_| {
                PolyBackendError::InvalidSmallMatrixArtifact("ring dimension overflows")
            })?;
        let bound_length = u32::try_from(bound_bytes.len())
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("bound width overflows"))?;
        let magnitude_width = u32::try_from(magnitude_bytes).map_err(|_| {
            PolyBackendError::InvalidSmallMatrixArtifact("coefficient width overflows")
        })?;
        let coefficient_count = u64::try_from(coefficient_count).map_err(|_| {
            PolyBackendError::InvalidSmallMatrixArtifact("coefficient count overflows")
        })?;
        let header_length = 4usize
            .checked_add(1)
            .and_then(|length| length.checked_add(8 * 3))
            .and_then(|length| length.checked_add(4))
            .and_then(|length| length.checked_add(bound_bytes.len()))
            .and_then(|length| length.checked_add(4 + 8))
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("header length overflows"))?;
        let mut bytes =
            Vec::with_capacity(header_length.checked_add(payload.len()).ok_or(
                PolyBackendError::InvalidSmallMatrixArtifact("artifact length overflows"),
            )?);
        bytes.extend_from_slice(SMALL_MATRIX_MAGIC);
        bytes.push(small_matrix_semantic_tag(semantic_kind));
        bytes.extend_from_slice(&rows.to_le_bytes());
        bytes.extend_from_slice(&columns.to_le_bytes());
        bytes.extend_from_slice(&ring_dimension.to_le_bytes());
        bytes.extend_from_slice(&bound_length.to_le_bytes());
        bytes.extend_from_slice(&bound_bytes);
        bytes.extend_from_slice(&magnitude_width.to_le_bytes());
        bytes.extend_from_slice(&coefficient_count.to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    fn small_matrix_from_bytes(
        &self,
        expected_schema: &ConcreteBoundedMatrixSchema,
        bytes: &[u8],
        expected_semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<M::SmallMatrix, Self::Error> {
        let (bound, magnitude_bytes, coefficient_count) = bounded_schema_parts(expected_schema)?;
        let mut offset = 0usize;
        if take_small_matrix_bytes(bytes, &mut offset, SMALL_MATRIX_MAGIC.len())? !=
            SMALL_MATRIX_MAGIC
        {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact("magic does not match"));
        }
        let semantic_kind = *take_small_matrix_bytes(bytes, &mut offset, 1)?
            .first()
            .expect("one-byte slice is nonempty");
        if semantic_kind != small_matrix_semantic_tag(expected_semantic_kind) {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                "semantic kind does not match",
            ));
        }
        let expected_rows = u64::try_from(expected_schema.matrix.rows)
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("row count overflows"))?;
        let expected_columns = u64::try_from(expected_schema.matrix.columns)
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("column count overflows"))?;
        let expected_ring_dimension = u64::try_from(expected_schema.matrix.ring_dimension)
            .map_err(|_| {
                PolyBackendError::InvalidSmallMatrixArtifact("ring dimension overflows")
            })?;
        if read_small_matrix_u64(bytes, &mut offset)? != expected_rows ||
            read_small_matrix_u64(bytes, &mut offset)? != expected_columns ||
            read_small_matrix_u64(bytes, &mut offset)? != expected_ring_dimension
        {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact("matrix shape does not match"));
        }
        let bound_length = usize::try_from(read_small_matrix_u32(bytes, &mut offset)?)
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("bound width overflows"))?;
        if bound_length != magnitude_bytes {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact("bound width does not match"));
        }
        let encoded_bound = take_small_matrix_bytes(bytes, &mut offset, bound_length)?;
        if BigUint::from_bytes_le(encoded_bound) != bound ||
            (bound.is_zero() && encoded_bound != [0]) ||
            (!bound.is_zero() && encoded_bound.last() == Some(&0))
        {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                "bound is not canonical or does not match",
            ));
        }
        let encoded_magnitude_bytes = usize::try_from(read_small_matrix_u32(bytes, &mut offset)?)
            .map_err(|_| {
            PolyBackendError::InvalidSmallMatrixArtifact("coefficient width overflows")
        })?;
        if encoded_magnitude_bytes != magnitude_bytes {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                "coefficient width does not match",
            ));
        }
        let encoded_coefficient_count = read_small_matrix_u64(bytes, &mut offset)?;
        let expected_coefficient_count = u64::try_from(coefficient_count).map_err(|_| {
            PolyBackendError::InvalidSmallMatrixArtifact("coefficient count overflows")
        })?;
        if encoded_coefficient_count != expected_coefficient_count {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                "coefficient count does not match",
            ));
        }
        let encoded_width = 1usize
            .checked_add(magnitude_bytes)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("coefficient width overflows"))?;
        let payload_length = coefficient_count
            .checked_mul(encoded_width)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("payload length overflows"))?;
        let payload = take_small_matrix_bytes(bytes, &mut offset, payload_length)?;
        if offset != bytes.len() {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact("artifact has trailing bytes"));
        }
        let parameters = self.parameters(&expected_schema.matrix)?;
        Ok(M::SmallMatrix::from_canonical_coefficients(
            parameters,
            expected_schema.matrix.rows,
            expected_schema.matrix.columns,
            bound,
            payload,
        )?)
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
    use mxx_primitives::poly::{PolyParams, dcrt::poly::DCRTPoly};

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

    #[test]
    fn decomposed_hash_uses_the_explicit_backend_layout() {
        let parameters = DCRTPolyParams::new(4, 1, 10, 5);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let digits = parameters.modulus_digits();
        let base = BigInt::from(1u8) << parameters.base_bits();
        let plain_type =
            ConcreteMatrixType { modulus: modulus.clone(), ring_dimension: 4, rows: 2, columns: 3 };
        let decomposed_type =
            ConcreteMatrixType { rows: plain_type.rows * digits, ..plain_type.clone() };
        let key = [7u8; 32];
        let tag = b"runtime-explicit-layout";
        let mut backend = cpu_backend([parameters]);

        let plain = backend.sample_hash(&plain_type, key, tag).expect("plain hash");
        let decomposed = backend
            .sample_hash_decomposed(&decomposed_type, key, tag, &base, digits)
            .expect("decomposed hash");
        let small_decomposed = backend
            .sample_hash_small_decomposed(&decomposed_type, key, tag, &base, digits)
            .expect("small decomposed hash");

        assert_eq!(decomposed.value(), &plain.decompose());
        assert_eq!(small_decomposed.value(), &plain.small_decompose());

        let gadget = DCRTPolyMatrix::gadget_matrix(decomposed.value().params(), plain_type.rows);
        assert_eq!(
            backend
                .multiply_small_rhs(&gadget, &decomposed)
                .expect("multiply compact regular decomposition"),
            plain
        );
    }

    #[test]
    fn compact_artifact_codec_keeps_semantics_external_and_rejects_malformed_payloads() {
        let parameters = DCRTPolyParams::new(4, 1, 16, 8);
        let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
        let digits = parameters.modulus_digits();
        let base = BigInt::from(1u8) << parameters.base_bits();
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType { modulus, ring_dimension: 4, rows: digits, columns: 2 },
            max_coefficient_bound: &base - BigInt::one(),
        };
        let mut backend = cpu_backend([parameters]);
        let value = backend
            .sample_hash_small_decomposed(
                &schema.matrix,
                [9u8; 32],
                b"compact-codec",
                &base,
                digits,
            )
            .expect("compact hash decomposition");

        let generic = backend
            .small_matrix_to_bytes(&value, &schema, SmallMatrixSemanticKind::Generic)
            .expect("encode generic compact matrix");
        let preimage = backend
            .small_matrix_to_bytes(&value, &schema, SmallMatrixSemanticKind::Preimage)
            .expect("encode preimage compact matrix");
        assert_eq!(&generic[..4], SMALL_MATRIX_MAGIC);
        assert_eq!(generic[4], 0);
        assert_eq!(preimage[4], 1);
        assert_eq!(generic[5..], preimage[5..]);
        assert_eq!(
            backend
                .small_matrix_from_bytes(&schema, &generic, SmallMatrixSemanticKind::Generic,)
                .expect("decode generic compact matrix"),
            value
        );
        assert!(
            backend
                .small_matrix_from_bytes(&schema, &generic, SmallMatrixSemanticKind::Preimage,)
                .is_err()
        );

        let mut trailing = generic.clone();
        trailing.push(0);
        assert!(
            backend
                .small_matrix_from_bytes(&schema, &trailing, SmallMatrixSemanticKind::Generic,)
                .is_err()
        );

        assert_eq!(schema.max_coefficient_bound, BigInt::from(255u16));
        let bound_width = 1usize;
        let payload_offset = 45 + bound_width;
        let mut negative_zero = generic;
        negative_zero[payload_offset] = 2;
        negative_zero[payload_offset + 1] = 0;
        assert!(
            backend
                .small_matrix_from_bytes(&schema, &negative_zero, SmallMatrixSemanticKind::Generic,)
                .is_err()
        );
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use crate::backend::poly_gpu::{GpuDcrtBackend, gpu_backend, gpu_backend_on};
}
