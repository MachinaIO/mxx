use super::{
    Backend, HashSampleRequest, IndexRange, MatrixMulAccumulateRequest, PreimageRequest,
    PreimageTarget, SampleRange, UniformSampleRequest,
};
use mxx_ir_core::{
    ParamEnv,
    node::{ConcatAxis, ConstantMatrix, HashVariant},
    types::ConcreteMatrixType,
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixColumnSource, PolyMatrixSmallRhs, SmallMatrixError, SmallPolyMatrix,
        dcrt_poly::DCRTPolyMatrix,
    },
    poly::{Poly, PolyParams, dcrt::params::DCRTPolyParams},
    sampler::{
        DistType, PolyHashSampler, PolyTrapdoorSampler, PolyUniformSampler,
        bounds::default_preimage_cutoff, hash::DCRTPolyHashSampler,
        trapdoor::DCRTPolyTrapdoorSampler, uniform::DCRTPolyUniformSampler,
    },
};
use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};
use thiserror::Error;

#[cfg(test)]
use mxx_primitives::sampler::bounds::matrix_within_coefficient_bound;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RingKey {
    pub modulus: BigInt,
    pub ring_dimension: usize,
}

#[derive(Debug, Error)]
pub enum PolyBackendError {
    #[error(transparent)]
    SmallMatrix(#[from] SmallMatrixError),
    #[error("invalid small-matrix artifact: {0}")]
    InvalidSmallMatrixArtifact(&'static str),
    #[error("GPU fleet calibration failed: {0}")]
    GpuCalibration(String),
    #[error("the requested GPU placement is unavailable through a direct device or peer copy")]
    UnsupportedPlacement,
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
    #[error("requested preimage bound {requested} is below minimum {minimum}")]
    PreimageBoundTooSmall { requested: BigInt, minimum: BigInt },
    #[error("preimage sampling exhausted {attempts} attempts for bound {bound}")]
    PreimageSamplingExhausted { attempts: usize, bound: BigInt },
    #[cfg(test)]
    #[error("injected release-fence failure")]
    InjectedReleaseFenceFailure,
}

pub struct PolyBackend<M, U, H, T>
where
    M: PolyMatrix,
{
    pub(super) parameters: Vec<BTreeMap<RingKey, <M::P as Poly>::Params>>,
    pub(super) active_placement: usize,
    pub(super) preimage_batch_calls: usize,
    #[cfg(test)]
    pub(super) matrix_serialization_batch_calls: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    pub(super) uniform_sampling_batch_calls: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    pub(super) hash_sampling_batch_calls: std::sync::atomic::AtomicUsize,
    pub(super) uniform_batch_dispatch:
        Option<fn(&mut Self, Vec<UniformSampleRequest>) -> Result<Vec<M>, PolyBackendError>>,
    pub(super) hash_batch_dispatch:
        Option<fn(&mut Self, Vec<HashSampleRequest>) -> Result<Vec<M>, PolyBackendError>>,
    #[cfg(test)]
    pub(super) preimage_batch_sizes: Vec<usize>,
    #[cfg(test)]
    pub(super) multiply_calls: usize,
    #[cfg(test)]
    pub(super) multiply_batch_sizes: Vec<usize>,
    #[cfg(test)]
    pub(super) multiply_small_rhs_batch_sizes: Vec<usize>,
    #[cfg(test)]
    pub(super) fail_next_release_fence: bool,
    pub(super) _marker: PhantomData<(M, U, H, T)>,
}

#[cfg(test)]
fn sample_bounded_candidate<M: PolyMatrix>(
    max_coefficient_bound: &BigUint,
    sample: impl FnMut() -> M,
) -> Result<M, PolyBackendError> {
    const MAX_ATTEMPTS: usize = 64;
    let mut sample = sample;
    for attempts in 1..=MAX_ATTEMPTS {
        let candidate = sample();
        if matrix_within_coefficient_bound(&candidate, max_coefficient_bound) {
            return Ok(candidate);
        }
        if attempts == MAX_ATTEMPTS {
            return Err(PolyBackendError::PreimageSamplingExhausted {
                attempts,
                bound: BigInt::from_biguint(Sign::Plus, max_coefficient_bound.clone()),
            });
        }
    }
    unreachable!("bounded sampling loop always returns")
}

pub(crate) trait CrtRecomposeMatrix: PolyMatrix {
    /// Reconstructs a full-modulus matrix from one-row plaintext levels.
    ///
    /// Each coefficient of level `i` is first nearest-scaled from the full
    /// modulus `q` into its plaintext modulus `p_i`:
    /// `round(p_i * y / q) mod p_i`.  The rounded level is then embedded in
    /// the full ring and multiplied by its reconstruction coefficient.  The
    /// operation is coefficient-wise; it is not a polynomial-wide
    /// approximation or a post-hoc correction.
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
                        // The `q / 2` term implements nearest-integer scaling
                        // for the canonical non-negative representative `y`.
                        // Reducing after division gives the residue in the
                        // plaintext level, including the wrap-around case for
                        // centered negative errors represented modulo `q`.
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
            #[cfg(test)]
            matrix_serialization_batch_calls: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            uniform_sampling_batch_calls: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            hash_sampling_batch_calls: std::sync::atomic::AtomicUsize::new(0),
            uniform_batch_dispatch: None,
            hash_batch_dispatch: None,
            #[cfg(test)]
            preimage_batch_sizes: Vec::new(),
            #[cfg(test)]
            multiply_calls: 0,
            #[cfg(test)]
            multiply_batch_sizes: Vec::new(),
            #[cfg(test)]
            multiply_small_rhs_batch_sizes: Vec::new(),
            #[cfg(test)]
            fail_next_release_fence: false,
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

    fn register(&mut self, parameters: <M::P as Poly>::Params) {
        self.register_at(self.active_placement, parameters);
    }

    pub(super) fn register_at(&mut self, placement: usize, parameters: <M::P as Poly>::Params) {
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

    /// Copies a complete matrix to the active GPU placement without host
    /// staging. Fleet-internal resharding must fail if direct or peer transfer
    /// is unavailable; silently staging through the CPU would invalidate the
    /// calibrated device-memory and transfer model.
    #[cfg(feature = "gpu")]
    pub(super) fn matrix_to_active_placement_peer_only(
        &self,
        value: &M,
    ) -> Result<M, PolyBackendError> {
        let target = self.parameters_for_matrix(value)?;
        if value.params() == target {
            return Ok(value.clone());
        }
        value.copy_to_params_direct(target).ok_or(PolyBackendError::UnsupportedPlacement)
    }

    #[cfg(test)]
    pub fn matrix_serialization_batch_calls(&self) -> usize {
        self.matrix_serialization_batch_calls.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn uniform_sampling_batch_calls(&self) -> usize {
        self.uniform_sampling_batch_calls.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn hash_sampling_batch_calls(&self) -> usize {
        self.hash_sampling_batch_calls.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Installs backend-specific sampling dispatchers while keeping the
    /// generic backend's scalar implementation as the safe default.
    #[cfg(feature = "gpu")]
    pub(super) fn set_sampling_batch_dispatch(
        &mut self,
        uniform: fn(&mut Self, Vec<UniformSampleRequest>) -> Result<Vec<M>, PolyBackendError>,
        hash: fn(&mut Self, Vec<HashSampleRequest>) -> Result<Vec<M>, PolyBackendError>,
    ) {
        self.uniform_batch_dispatch = Some(uniform);
        self.hash_batch_dispatch = Some(hash);
    }

    #[cfg(test)]
    pub(crate) fn preimage_batch_sizes(&self) -> &[usize] {
        &self.preimage_batch_sizes
    }

    #[cfg(test)]
    pub(crate) fn multiply_calls(&self) -> usize {
        self.multiply_calls
    }

    #[cfg(test)]
    pub(crate) fn multiply_small_rhs_batch_sizes(&self) -> &[usize] {
        &self.multiply_small_rhs_batch_sizes
    }

    #[cfg(test)]
    pub(crate) fn fail_next_release_fence(&mut self) {
        self.fail_next_release_fence = true;
    }

    pub(super) fn parameters(
        &self,
        matrix_type: &ConcreteMatrixType,
    ) -> Result<&<M::P as Poly>::Params, PolyBackendError> {
        self.parameters_at(self.active_placement, matrix_type)
    }

    pub(super) fn parameters_at(
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
        self.parameters_for_params(matrix.params())
    }

    fn parameters_for_params(
        &self,
        parameters: &<M::P as Poly>::Params,
    ) -> Result<&<M::P as Poly>::Params, PolyBackendError> {
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

    fn small_matrix_is_on_active_placement(&self, _value: &M::SmallMatrix) -> bool {
        self.parameters_for_params(_value.params()).is_ok_and(|target| _value.params() == target)
    }

    fn small_matrix_to_active_placement(
        &mut self,
        value: &M::SmallMatrix,
    ) -> Result<M::SmallMatrix, Self::Error> {
        let target = self.parameters_for_params(value.params())?.clone();
        if value.params() == &target {
            return Ok(value.clone());
        }

        // Transfer only the canonical compact owner.  In particular, do not
        // decode or materialize the O((log q)^2) RHS as a full DCRT matrix
        // merely to move it between placements.  Reconstructing the same
        // bounded owner under the target parameters preserves its complete
        // rows, columns, inclusive bound, and canonical payload semantics.
        let payload = value.to_canonical_coefficients()?;
        Ok(M::SmallMatrix::from_canonical_coefficients(
            &target,
            value.rows(),
            value.columns(),
            value.max_coefficient_bound().clone(),
            &payload,
        )?)
    }

    fn fence_released_memory(&mut self) -> Result<(), Self::Error> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_release_fence) {
            return Err(PolyBackendError::InjectedReleaseFenceFailure);
        }
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
        #[cfg(test)]
        {
            self.multiply_calls += 1;
        }
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
        #[cfg(test)]
        self.multiply_batch_sizes.push(inputs.len());
        Ok(M::multiply_batch_out_of_place(inputs))
    }

    fn multiply_small_rhs(&mut self, left: &M, right: &M::SmallMatrix) -> Result<M, Self::Error> {
        self.parameters_for_matrix(left)?;
        Ok(left.multiply_small_rhs(right)?)
    }

    fn multiply_small_rhs_batch(
        &mut self,
        inputs: Vec<(Arc<M>, Arc<M::SmallMatrix>)>,
    ) -> Result<Vec<M>, Self::Error> {
        #[cfg(test)]
        self.multiply_small_rhs_batch_sizes.push(inputs.len());
        inputs.into_iter().map(|(left, right)| self.multiply_small_rhs(&left, &right)).collect()
    }

    fn small_matrix_to_bytes(
        &self,
        value: &M::SmallMatrix,
        expected_schema: &mxx_ir_core::artifact::ConcreteBoundedMatrixSchema,
        _semantic_kind: mxx_ir_core::artifact::SmallMatrixSemanticKind,
    ) -> Result<Vec<u8>, Self::Error> {
        let parameters = self.parameters(&expected_schema.matrix)?;
        let bound = expected_schema
            .max_coefficient_bound
            .to_biguint()
            .ok_or(PolyBackendError::InvalidInteger)?;
        value.validate_metadata(
            parameters,
            expected_schema.matrix.rows,
            expected_schema.matrix.columns,
            &bound,
        )?;
        Ok(value.to_canonical_coefficients()?)
    }

    fn small_matrix_matches_schema(
        &self,
        value: &M::SmallMatrix,
        schema: &mxx_ir_core::artifact::ConcreteBoundedMatrixSchema,
    ) -> bool {
        let Ok(parameters) = self.parameters(&schema.matrix) else { return false };
        let Some(bound) = schema.max_coefficient_bound.to_biguint() else { return false };
        value
            .validate_metadata(parameters, schema.matrix.rows, schema.matrix.columns, &bound)
            .is_ok()
    }

    fn small_matrix_to_untyped_bytes(&self, value: &M::SmallMatrix) -> Vec<u8> {
        value.to_canonical_coefficients().unwrap_or_default()
    }

    fn small_matrix_from_bytes(
        &self,
        expected_schema: &mxx_ir_core::artifact::ConcreteBoundedMatrixSchema,
        bytes: &[u8],
        _expected_semantic_kind: mxx_ir_core::artifact::SmallMatrixSemanticKind,
    ) -> Result<M::SmallMatrix, Self::Error> {
        let params = self.parameters(&expected_schema.matrix)?;
        let bound = expected_schema
            .max_coefficient_bound
            .to_biguint()
            .ok_or(PolyBackendError::InvalidInteger)?;
        Ok(M::SmallMatrix::from_canonical_coefficients(
            params,
            expected_schema.matrix.rows,
            expected_schema.matrix.columns,
            bound,
            bytes,
        )?)
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

    fn ring_automorphism(&mut self, value: &M, index: usize) -> Result<M, Self::Error> {
        Ok(value.ring_automorphism_out_of_place(index))
    }

    fn ring_automorphism_batch(
        &mut self,
        inputs: Vec<(Arc<M>, usize)>,
    ) -> Result<Vec<M>, Self::Error> {
        Ok(M::ring_automorphism_batch_out_of_place(inputs))
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

    fn write_columns(
        &mut self,
        target: &mut M,
        offset: usize,
        columns: &[M],
    ) -> Result<(), Self::Error> {
        let (rows, target_columns) = target.size();
        if offset.checked_add(columns.len()).is_none_or(|end| end > target_columns) ||
            columns.iter().any(|column| column.size() != (rows, 1))
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let sources = columns.iter().collect::<Vec<_>>();
        target.copy_columns_from(&sources, offset);
        Ok(())
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

    fn sample_uniform_batch(
        &mut self,
        requests: Vec<UniformSampleRequest>,
    ) -> Result<Vec<M>, Self::Error> {
        #[cfg(test)]
        self.uniform_sampling_batch_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(dispatch) = self.uniform_batch_dispatch {
            return dispatch(self, requests);
        }
        requests
            .into_iter()
            .map(|request| self.sample_uniform(&request.matrix_type, &request.range))
            .collect()
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
        variant: HashVariant,
        gadget_layout: Option<(&BigInt, usize)>,
    ) -> Result<M, Self::Error> {
        let parameters = self.parameters(ty)?;
        let sampler = H::new();
        Ok(match variant {
            HashVariant::Plain => {
                if gadget_layout.is_some() {
                    return Err(PolyBackendError::InvalidInteger);
                }
                sampler.sample_hash(
                    parameters,
                    key,
                    tag,
                    ty.rows,
                    ty.columns,
                    DistType::FinRingDist,
                )
            }
            HashVariant::Decomposed => {
                let (base, digits) = gadget_layout.ok_or(PolyBackendError::InvalidInteger)?;
                self.validate_gadget_layout(ty, base, digits, false)?;
                if ty.rows % digits != 0 {
                    return Err(PolyBackendError::InvalidInteger);
                }
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
                let (base, digits) = gadget_layout.ok_or(PolyBackendError::InvalidInteger)?;
                self.validate_gadget_layout(ty, base, digits, true)?;
                if ty.rows % digits != 0 {
                    return Err(PolyBackendError::InvalidInteger);
                }
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

    fn sample_hash_small(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        variant: HashVariant,
        gadget_layout: (&BigInt, usize),
    ) -> Result<M::SmallMatrix, Self::Error> {
        let parameters = self.parameters(ty)?;
        let (base, digits) = gadget_layout;
        let small = matches!(variant, HashVariant::SmallDecomposed);
        if !matches!(variant, HashVariant::Decomposed | HashVariant::SmallDecomposed) {
            return Err(PolyBackendError::InvalidInteger);
        }
        self.validate_gadget_layout(ty, base, digits, small)?;
        if ty.rows % digits != 0 {
            return Err(PolyBackendError::InvalidInteger);
        }
        // Keep the hash source in coefficient form and decompose directly
        // into the compact owner.  This avoids first materializing the full
        // expanded gadget matrix (which is the dominant GPU peak).
        let source = H::new().sample_hash_gadget_source(
            parameters,
            key,
            tag,
            ty.rows / digits,
            ty.columns,
            DistType::FinRingDist,
        );
        Ok(source.gadget_decompose(small)?)
    }

    fn sample_hash_small_batch(
        &mut self,
        requests: Vec<HashSampleRequest>,
    ) -> Result<Vec<M::SmallMatrix>, Self::Error> {
        requests
            .into_iter()
            .map(|request| {
                let layout = request.gadget_layout.ok_or(PolyBackendError::InvalidInteger)?;
                self.sample_hash_small(
                    &request.matrix_type,
                    request.key,
                    &request.tag,
                    request.variant,
                    (&layout.0, layout.1),
                )
            })
            .collect()
    }

    fn sample_hash_batch(
        &mut self,
        requests: Vec<HashSampleRequest>,
    ) -> Result<Vec<M>, Self::Error> {
        #[cfg(test)]
        self.hash_sampling_batch_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(dispatch) = self.hash_batch_dispatch {
            return dispatch(self, requests);
        }
        requests
            .into_iter()
            .map(|request| {
                let gadget_layout =
                    request.gadget_layout.as_ref().map(|(base, digits)| (base, *digits));
                self.sample_hash(
                    &request.matrix_type,
                    request.key,
                    &request.tag,
                    request.variant,
                    gadget_layout,
                )
            })
            .collect()
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
        target: &dyn PolyMatrixColumnSource<M>,
        randomness_seed: [u8; 32],
    ) -> Result<M::SmallMatrix, Self::Error> {
        self.validate_preimage_bound(ty, sigma, gadget_base, digit_count, max_coefficient_bound)?;
        let parameters = self.parameters(ty)?;
        Self::validate_regular_gadget_layout(parameters, gadget_base, digit_count)?;
        let max_coefficient_bound =
            max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
        let sampler = T::new(parameters, sigma);
        Ok(sampler.preimage(
            parameters,
            trapdoor,
            public,
            target,
            max_coefficient_bound,
            randomness_seed,
        )?)
    }

    fn preimage_target(
        &mut self,
        value: Arc<M>,
    ) -> Result<(Arc<dyn PolyMatrixColumnSource<M>>, Arc<Vec<u8>>), Self::Error> {
        let rows = value.row_size();
        let columns = value.col_size();
        let params = value.params().clone();
        let bytes = Arc::new(
            Arc::try_unwrap(value)
                .map(|value| value.into_cpu_staging_bytes())
                .unwrap_or_else(|value| value.as_ref().to_cpu_staging_bytes()),
        );
        Ok((Arc::new(PreimageTarget::staged(params, rows, columns, bytes.clone())), bytes))
    }

    fn matrix_from_cpu_staging_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<M, Self::Error> {
        Ok(M::from_cpu_staging_bytes(self.parameters(ty)?, bytes))
    }

    fn preimage_target_from_staging(
        &self,
        ty: &ConcreteMatrixType,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Result<Arc<dyn PolyMatrixColumnSource<M>>, Self::Error> {
        let params = self.parameters(ty)?.clone();
        Ok(Arc::new(PreimageTarget::staged(params, rows, columns, bytes)))
    }

    fn validate_preimage_bound(
        &self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
    ) -> Result<(), Self::Error> {
        let parameters = self.parameters(ty)?;
        let base = gadget_base.to_u32().ok_or(PolyBackendError::InvalidInteger)?;
        let public_rows = ty
            .rows
            .checked_div(digit_count.checked_add(2).ok_or(PolyBackendError::InvalidInteger)?)
            .filter(|rows| *rows > 0)
            .ok_or(PolyBackendError::InvalidInteger)?;
        let minimum = default_preimage_cutoff(
            parameters.ring_dimension(),
            public_rows,
            parameters.modulus_digits(),
            base,
            sigma,
        )
        .ok_or(PolyBackendError::InvalidInteger)?;
        let requested =
            max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?;
        if requested < minimum {
            return Err(PolyBackendError::PreimageBoundTooSmall {
                requested: max_coefficient_bound.clone(),
                minimum: BigInt::from_biguint(Sign::Plus, minimum),
            });
        }
        Ok(())
    }

    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<M, T::Trapdoor>>,
    ) -> Result<Vec<M::SmallMatrix>, Self::Error> {
        for request in &requests {
            self.validate_preimage_bound(
                &request.matrix_type,
                request.sigma,
                &request.gadget_base,
                request.digit_count,
                &request.max_coefficient_bound,
            )?;
        }
        self.preimage_batch_calls += 1;
        #[cfg(test)]
        self.preimage_batch_sizes.push(requests.len());
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
                        &request.max_coefficient_bound,
                        &request.trapdoor,
                        &request.public,
                        request.target.as_ref(),
                        request.randomness_seed,
                    )
                })
                .collect()
        }
        #[cfg(feature = "gpu")]
        {
            super::poly_gpu::sample_preimage_batch(self, requests)
        }
    }

    fn sample_preimage_batches_by_placement(
        &mut self,
        batches: Vec<(usize, Vec<PreimageRequest<M, T::Trapdoor>>)>,
    ) -> Result<Vec<(usize, Vec<M::SmallMatrix>)>, Self::Error> {
        for (_, requests) in &batches {
            for request in requests {
                self.validate_preimage_bound(
                    &request.matrix_type,
                    request.sigma,
                    &request.gadget_base,
                    request.digit_count,
                    &request.max_coefficient_bound,
                )?;
            }
        }
        self.preimage_batch_calls += batches.len();
        #[cfg(test)]
        self.preimage_batch_sizes.extend(batches.iter().map(|(_, requests)| requests.len()));
        #[cfg(feature = "gpu")]
        {
            super::poly_gpu::sample_preimage_batches_by_placement(self, batches)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let original = self.active_placement;
            let result = batches
                .into_iter()
                .map(|(placement, requests)| {
                    self.active_placement = placement;
                    requests
                        .into_iter()
                        .map(|request| {
                            self.sample_preimage(
                                &request.matrix_type,
                                request.sigma,
                                &request.gadget_base,
                                request.digit_count,
                                &request.max_coefficient_bound,
                                request.trapdoor.as_ref(),
                                request.public.as_ref(),
                                request.target.as_ref(),
                                request.randomness_seed,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map(|outputs| (placement, outputs))
                })
                .collect();
            self.active_placement = original;
            result
        }
    }

    fn gadget_decompose(&mut self, value: &M, small: bool) -> Result<M::SmallMatrix, Self::Error> {
        Ok(value.clone().gadget_decompose(small)?)
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
        #[cfg(test)]
        {
            // The counter is test-only instrumentation for verifying executor wave batching.
            // It does not participate in serialization or production state.
            self.matrix_serialization_batch_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
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
    use mxx_primitives::{
        matrix::{CpuSmallMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::poly::DCRTPoly},
    };

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
    fn bounded_candidate_sampling_retries_without_clipping() {
        let parameters = DCRTPolyParams::new(2, 1, 10, 5);
        let candidate = |value: u8| {
            DCRTPolyMatrix::from_poly_vec_row(
                &parameters,
                vec![DCRTPoly::from_biguint_to_constant(&parameters, BigUint::from(value))],
            )
        };
        let rejected = candidate(7);
        let accepted = candidate(2);
        let mut draws = 0;
        let sampled = sample_bounded_candidate(&BigUint::from(2u8), || {
            draws += 1;
            if draws == 1 { rejected.clone() } else { accepted.clone() }
        })
        .expect("bounded candidate");
        assert_eq!(draws, 2);
        assert_eq!(sampled, accepted);
    }

    #[test]
    fn compact_small_matrix_moves_between_parameter_placements_without_shape_loss() {
        let source_parameters = DCRTPolyParams::new(4, 1, 10, 2);
        // The modulus and ring are identical, while the placement-specific
        // gadget base differs. This forces the compact-owner transfer path
        // without changing the logical matrix schema.
        let target_parameters = DCRTPolyParams::new(4, 1, 10, 5);
        let source_matrix = DCRTPolyMatrix::from_poly_vec(
            &source_parameters,
            vec![
                vec![
                    DCRTPoly::from_biguints(
                        &source_parameters,
                        &[
                            BigUint::from(1u8),
                            BigUint::from(2u8),
                            BigUint::from(3u8),
                            BigUint::from(4u8),
                        ],
                    ),
                    DCRTPoly::from_biguints(
                        &source_parameters,
                        &[
                            BigUint::from(5u8),
                            BigUint::from(6u8),
                            BigUint::from(7u8),
                            BigUint::from(8u8),
                        ],
                    ),
                    DCRTPoly::from_biguints(
                        &source_parameters,
                        &[
                            BigUint::from(9u8),
                            BigUint::from(10u8),
                            BigUint::from(11u8),
                            BigUint::from(12u8),
                        ],
                    ),
                ],
                vec![
                    DCRTPoly::from_biguints(
                        &source_parameters,
                        &[
                            BigUint::from(13u8),
                            BigUint::from(14u8),
                            BigUint::from(15u8),
                            BigUint::from(16u8),
                        ],
                    ),
                    DCRTPoly::from_biguints(
                        &source_parameters,
                        &[
                            BigUint::from(17u8),
                            BigUint::from(18u8),
                            BigUint::from(19u8),
                            BigUint::from(20u8),
                        ],
                    ),
                    DCRTPoly::from_biguints(
                        &source_parameters,
                        &[
                            BigUint::from(21u8),
                            BigUint::from(22u8),
                            BigUint::from(23u8),
                            BigUint::from(24u8),
                        ],
                    ),
                ],
            ],
        );
        let owner =
            CpuSmallMatrix::new(source_matrix, BigUint::from(24u8)).expect("source compact owner");
        let source_payload = owner.to_canonical_coefficients().expect("source payload");
        let mut backend = CpuDcrtBackend::default();
        backend.parameters.push(BTreeMap::new());
        backend.register_at(0, source_parameters.clone());
        backend.register_at(1, target_parameters.clone());

        assert!(backend.set_active_placement(0));
        assert!(backend.small_matrix_is_on_active_placement(&owner));
        assert!(backend.set_active_placement(1));
        assert!(!backend.small_matrix_is_on_active_placement(&owner));
        let moved =
            backend.small_matrix_to_active_placement(&owner).expect("compact placement transfer");

        assert_eq!(moved.params(), &target_parameters);
        assert_eq!(moved.size(), owner.size());
        assert_eq!(moved.max_coefficient_bound(), owner.max_coefficient_bound());
        assert_eq!(moved.to_canonical_coefficients().expect("target payload"), source_payload);
    }

    #[test]
    fn preimage_cutoff_accepts_authoritative_minimum() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let digit_count = parameters.modulus_digits();
        let matrix_type = ConcreteMatrixType {
            modulus: BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone()),
            ring_dimension: parameters.ring_dimension() as usize,
            rows: digit_count + 2,
            columns: 1,
        };
        let backend = cpu_backend([parameters]);
        let minimum = default_preimage_cutoff(
            matrix_type.ring_dimension as u32,
            1,
            digit_count,
            1u32 << 4,
            5.0,
        )
        .expect("cutoff");
        let minimum = BigInt::from_biguint(Sign::Plus, minimum);
        assert!(
            backend
                .validate_preimage_bound(
                    &matrix_type,
                    5.0,
                    &BigInt::from(1u32 << 4),
                    digit_count,
                    &minimum,
                )
                .is_ok()
        );
        let below = &minimum - 1;
        assert!(matches!(
            backend.validate_preimage_bound(
                &matrix_type,
                5.0,
                &BigInt::from(1u32 << 4),
                digit_count,
                &below,
            ),
            Err(PolyBackendError::PreimageBoundTooSmall { .. })
        ));
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

        let plain = backend
            .sample_hash(&plain_type, key, tag, HashVariant::Plain, None)
            .expect("plain hash");
        let decomposed = backend
            .sample_hash(&decomposed_type, key, tag, HashVariant::Decomposed, Some((&base, digits)))
            .expect("decomposed hash");
        let small_decomposed = backend
            .sample_hash(
                &decomposed_type,
                key,
                tag,
                HashVariant::SmallDecomposed,
                Some((&base, digits)),
            )
            .expect("small decomposed hash");

        assert_eq!(decomposed, plain.decompose());
        assert_eq!(small_decomposed, plain.small_decompose());

        let expected_decomposed = backend
            .sample_hash_small(&decomposed_type, key, tag, HashVariant::Decomposed, (&base, digits))
            .expect("decomposed owner");
        let expected_small = backend
            .sample_hash_small(
                &decomposed_type,
                key,
                tag,
                HashVariant::SmallDecomposed,
                (&base, digits),
            )
            .expect("small owner");
        let batched = backend
            .sample_hash_small_batch(vec![
                HashSampleRequest {
                    matrix_type: decomposed_type.clone(),
                    key,
                    tag: tag.to_vec(),
                    variant: HashVariant::Decomposed,
                    gadget_layout: Some((base.clone(), digits)),
                },
                HashSampleRequest {
                    matrix_type: decomposed_type.clone(),
                    key,
                    tag: tag.to_vec(),
                    variant: HashVariant::SmallDecomposed,
                    gadget_layout: Some((base.clone(), digits)),
                },
            ])
            .expect("batched compact hashes");
        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0], expected_decomposed);
        assert_eq!(batched[1], expected_small);
    }

    #[test]
    fn crt_recompose_rounds_each_coefficient_before_reconstruction() {
        let parameters = DCRTPolyParams::new(8, 2, 17, 8);
        let q = parameters.modulus().as_ref().clone();
        let q_int = BigInt::from_biguint(Sign::Plus, q.clone());
        let plaintext_moduli = [BigInt::from(3u8), BigInt::from(5u8)];
        let reconstruction_coefficients = [BigInt::from(1u8), BigInt::from(-2i8)];

        // These errors exercise zero, both signs, and values immediately below
        // the nearest-rounding half interval.  A centered negative value is
        // represented by its canonical residue modulo q, just as it is in a
        // runtime DCRT polynomial.
        let error = |plaintext_modulus: &BigInt, denominator: u8, negative: bool| {
            let denominator = plaintext_modulus * denominator;
            let magnitude = (&q_int / denominator) - 1u8;
            if negative { -magnitude } else { magnitude }
        };
        let make_level = |plaintext_modulus: &BigInt| {
            (0..2)
                .map(|column| {
                    (0..parameters.ring_dimension() as usize)
                        .map(|coefficient| {
                            let z = BigInt::from((column + coefficient) as u8) % plaintext_modulus;
                            let signed = (&q_int * &z) / plaintext_modulus +
                                match coefficient {
                                    0 => BigInt::zero(),
                                    1 => error(plaintext_modulus, 4, false),
                                    2 => error(plaintext_modulus, 4, true),
                                    3 => error(plaintext_modulus, 2, false),
                                    _ => error(plaintext_modulus, 2, true),
                                };
                            let canonical = ((signed % &q_int) + &q_int) % &q_int;
                            canonical.to_biguint().expect("canonical coefficient is nonnegative")
                        })
                        .collect::<Vec<_>>()
                })
                .map(|coefficients| DCRTPoly::from_biguints(&parameters, &coefficients))
                .collect::<Vec<_>>()
        };

        let levels = plaintext_moduli
            .iter()
            .map(make_level)
            .map(|polynomials| DCRTPolyMatrix::from_poly_vec_row(&parameters, polynomials))
            .collect::<Vec<_>>();

        let expected_levels = plaintext_moduli
            .iter()
            .zip(&levels)
            .map(|(plaintext_modulus, level)| {
                let rounded = (0..level.col_size())
                    .map(|column| {
                        let coefficients = level
                            .entry(0, column)
                            .coeffs_biguints()
                            .into_iter()
                            .map(|value| {
                                let value = BigInt::from_biguint(Sign::Plus, value);
                                (((plaintext_modulus * value + &q_int / 2u8) / &q_int) %
                                    plaintext_modulus)
                                    .to_biguint()
                                    .expect("rounded coefficient is nonnegative")
                            })
                            .collect::<Vec<_>>();
                        DCRTPoly::from_biguints(&parameters, &coefficients)
                    })
                    .collect::<Vec<_>>();
                DCRTPolyMatrix::from_poly_vec_row(&parameters, rounded)
            })
            .collect::<Vec<_>>();
        let mut expected = DCRTPolyMatrix::zero(&parameters, 1, 2);
        for (level, coefficient) in expected_levels.iter().zip(&reconstruction_coefficients) {
            let residue = ((coefficient % &q_int) + &q_int) % &q_int;
            let scalar = DCRTPoly::from_biguint_to_constant(
                &parameters,
                residue.to_biguint().expect("reconstruction residue is nonnegative"),
            );
            expected.add_in_place(&(level * scalar));
        }

        let actual = crt_recompose_cpu(&levels, &plaintext_moduli, &reconstruction_coefficients)
            .expect("CRT recomposition");
        assert_eq!(actual, expected);
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    pub use crate::backend::poly_gpu::{GpuDcrtBackend, gpu_backend, gpu_backend_on};
}
