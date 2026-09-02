use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix},
    types::ConcreteMatrixType,
};
use num_bigint::BigInt;
use std::{fmt::Debug, sync::Arc};

pub mod poly;
#[cfg(feature = "gpu")]
pub mod poly_gpu;

#[derive(Clone, Debug)]
pub struct PreimageRequest<M, T> {
    pub matrix_type: ConcreteMatrixType,
    pub sigma: f64,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub max_coefficient_bound: BigInt,
    pub trapdoor: Arc<T>,
    pub public: Arc<M>,
    pub target: Arc<M>,
}

#[derive(Clone, Debug)]
pub struct MatrixMulAccumulateRequest<M> {
    pub products: Vec<(BigInt, Arc<M>, Arc<M>)>,
    pub bias: Option<Arc<M>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleRange {
    pub minimum: BigInt,
    pub maximum: BigInt,
}

pub trait Backend {
    type Matrix: Clone + Debug + PartialEq + Send + Sync;
    type SmallMatrix: Clone + Debug + PartialEq + Send + Sync;
    type Trapdoor: Clone + Debug + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    fn placement_count(&self) -> usize {
        1
    }
    fn active_placement(&self) -> usize {
        0
    }
    fn set_active_placement(&mut self, placement: usize) -> bool {
        placement == 0
    }
    fn matrix_to_active_placement(
        &mut self,
        value: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Ok(value.clone())
    }
    fn matrix_is_on_active_placement(&self, _value: &Self::Matrix) -> bool {
        true
    }
    fn small_matrix_to_active_placement(
        &mut self,
        value: &Self::SmallMatrix,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Ok(value.clone())
    }
    fn small_matrix_is_on_active_placement(&self, _value: &Self::SmallMatrix) -> bool {
        true
    }
    /// Waits only for releases queued on backend-owned release streams.
    fn fence_released_memory(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn matrix_to_placements(
        &mut self,
        value: &Self::Matrix,
    ) -> Result<Vec<Option<Self::Matrix>>, Self::Error> {
        let original = self.active_placement();
        let result = (|| {
            let mut placed = Vec::with_capacity(self.placement_count());
            for placement in 0..self.placement_count() {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                placed.push(if self.matrix_is_on_active_placement(value) {
                    None
                } else {
                    Some(self.matrix_to_active_placement(value)?)
                });
            }
            Ok(placed)
        })();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }
    fn small_matrix_to_placements(
        &mut self,
        value: &Self::SmallMatrix,
    ) -> Result<Vec<Option<Self::SmallMatrix>>, Self::Error> {
        let original = self.active_placement();
        let result = (|| {
            let mut placed = Vec::with_capacity(self.placement_count());
            for placement in 0..self.placement_count() {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                placed.push(if self.small_matrix_is_on_active_placement(value) {
                    None
                } else {
                    Some(self.small_matrix_to_active_placement(value)?)
                });
            }
            Ok(placed)
        })();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }
    fn trapdoor_to_active_placement(
        &mut self,
        _ty: &ConcreteMatrixType,
        value: &Self::Trapdoor,
    ) -> Result<Self::Trapdoor, Self::Error> {
        Ok(value.clone())
    }
    fn trapdoor_to_placements(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &Self::Trapdoor,
        source_placement: usize,
    ) -> Result<Vec<Self::Trapdoor>, Self::Error> {
        let original = self.active_placement();
        let result = (|| {
            let mut placed = Vec::with_capacity(self.placement_count());
            for placement in 0..self.placement_count() {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                placed.push(if placement == source_placement {
                    value.clone()
                } else {
                    self.trapdoor_to_active_placement(ty, value)?
                });
            }
            Ok(placed)
        })();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }

    fn constant_matrix(
        &mut self,
        ty: &ConcreteMatrixType,
        value: &ConstantMatrix,
        env: &ParamEnv,
    ) -> Result<Self::Matrix, Self::Error>;
    fn add(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn add_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(left, right)| self.add(&left, &right)).collect()
    }
    fn sub(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sub_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(left, right)| self.sub(&left, &right)).collect()
    }
    fn multiply(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn multiply_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(left, right)| self.multiply(&left, &right)).collect()
    }
    fn matrix_mul_accumulate(
        &mut self,
        request: MatrixMulAccumulateRequest<Self::Matrix>,
    ) -> Result<Self::Matrix, Self::Error> {
        let mut products = request.products.into_iter();
        let (coefficient, left, right) =
            products.next().expect("validated multi-row GEMM has a product");
        let product = self.multiply(&left, &right)?;
        let mut output = self.scale_integer(&product, &coefficient)?;
        for (coefficient, left, right) in products {
            let product = self.multiply(&left, &right)?;
            let product = self.scale_integer(&product, &coefficient)?;
            output = self.add(&output, &product)?;
        }
        if let Some(bias) = request.bias {
            output = self.add(&output, &bias)?;
        }
        Ok(output)
    }
    fn matrix_mul_accumulate_batch(
        &mut self,
        requests: Vec<MatrixMulAccumulateRequest<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        requests.into_iter().map(|request| self.matrix_mul_accumulate(request)).collect()
    }
    fn negate(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error>;
    fn negate_batch(
        &mut self,
        inputs: Vec<Arc<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|value| self.negate(&value)).collect()
    }
    fn scale_integer(
        &mut self,
        value: &Self::Matrix,
        scalar: &BigInt,
    ) -> Result<Self::Matrix, Self::Error>;
    fn scale_integer_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, BigInt)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(value, scalar)| self.scale_integer(&value, &scalar)).collect()
    }
    fn transpose(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error>;
    fn slice(
        &mut self,
        value: &Self::Matrix,
        rows: Option<&IndexRange>,
        columns: Option<&IndexRange>,
    ) -> Result<Self::Matrix, Self::Error>;
    fn tensor(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn concat(
        &mut self,
        inputs: &[&Self::Matrix],
        axis: ConcatAxis,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_uniform(
        &mut self,
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_gaussian(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: &BigInt,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_hash_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn sample_hash_small_decomposed(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn sample_trapdoor(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
    ) -> Result<(Self::Matrix, Self::Trapdoor), Self::Error>;
    fn sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &Self::Matrix,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<Self::Matrix, Self::Trapdoor>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
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
                )
            })
            .collect()
    }
    fn sample_preimage_batches_by_placement(
        &mut self,
        batches: Vec<(usize, Vec<PreimageRequest<Self::Matrix, Self::Trapdoor>>)>,
    ) -> Result<Vec<(usize, Vec<Self::SmallMatrix>)>, Self::Error> {
        let original = self.active_placement();
        let result = batches
            .into_iter()
            .map(|(placement, requests)| {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                self.sample_preimage_batch(requests).map(|outputs| (placement, outputs))
            })
            .collect();
        assert!(self.set_active_placement(original), "backend rejected its active placement");
        result
    }
    fn validate_gadget_layout(
        &self,
        _ty: &ConcreteMatrixType,
        _gadget_base: &BigInt,
        _digit_count: usize,
        _small: bool,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn gadget_decompose(
        &mut self,
        value: &Self::Matrix,
        small: bool,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn multiply_small_rhs(
        &mut self,
        lhs: &Self::Matrix,
        rhs: &Self::SmallMatrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn extract_coefficient(
        &mut self,
        value: &Self::Matrix,
        position: usize,
    ) -> Result<BigInt, Self::Error>;
    fn threshold_decode(
        &mut self,
        value: &Self::Matrix,
        plaintext_modulus: &BigInt,
        length: usize,
    ) -> Result<Vec<BigInt>, Self::Error>;
    fn pack_polynomial_coefficients(
        &mut self,
        ty: &ConcreteMatrixType,
        bits: &[bool],
        coefficient_bits: usize,
    ) -> Result<Self::Matrix, Self::Error>;
    fn crt_recompose(
        &mut self,
        levels: &[Self::Matrix],
        plaintext_moduli: &[BigInt],
        reconstruction_coefficients: &[BigInt],
    ) -> Result<Self::Matrix, Self::Error>;

    fn matrix_to_bytes(&self, value: &Self::Matrix) -> Vec<u8>;
    fn matrices_to_bytes(&self, values: &[&Self::Matrix]) -> Vec<Vec<u8>> {
        values.iter().map(|value| self.matrix_to_bytes(value)).collect()
    }
    fn matrix_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error>;
    fn small_matrix_to_bytes(
        &self,
        value: &Self::SmallMatrix,
        expected_schema: &ConcreteBoundedMatrixSchema,
        semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Vec<u8>, Self::Error>;
    fn small_matrix_from_bytes(
        &self,
        expected_schema: &ConcreteBoundedMatrixSchema,
        bytes: &[u8],
        expected_semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn trapdoor_to_bytes(&self, value: &Self::Trapdoor) -> Vec<u8>;
    fn trapdoor_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Trapdoor, Self::Error>;
}

pub enum RuntimeValue<B: Backend> {
    Int(BigInt),
    Real(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    TypedBlob(Vec<u8>),
    Matrix(Arc<B::Matrix>),
    SmallMatrix(Arc<B::SmallMatrix>),
    Trapdoor {
        secret: Option<Arc<B::Trapdoor>>,
        public: Arc<B::Matrix>,
        matrix_type: ConcreteMatrixType,
        sigma: f64,
        gadget_base: BigInt,
        digit_count: usize,
        gadget_small: Option<bool>,
    },
    LazyArtifact {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        index: Option<usize>,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    LazyArtifactFamily {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    StagedArtifact {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        index: usize,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    StagedArtifactFamily {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        descriptor: mxx_ir_core::artifact::ManifestArtifact,
    },
    IndexedFamily(Vec<RuntimeValue<B>>),
}

impl<B: Backend> Clone for RuntimeValue<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Int(value) => Self::Int(value.clone()),
            Self::Real(value) => Self::Real(*value),
            Self::Bool(value) => Self::Bool(*value),
            Self::Bytes(value) => Self::Bytes(value.clone()),
            Self::TypedBlob(value) => Self::TypedBlob(value.clone()),
            Self::Matrix(value) => Self::Matrix(value.clone()),
            Self::SmallMatrix(value) => Self::SmallMatrix(value.clone()),
            Self::Trapdoor {
                secret,
                public,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
                gadget_small,
            } => Self::Trapdoor {
                secret: secret.clone(),
                public: public.clone(),
                matrix_type: matrix_type.clone(),
                sigma: *sigma,
                gadget_base: gadget_base.clone(),
                digit_count: *digit_count,
                gadget_small: *gadget_small,
            },
            Self::LazyArtifact { production, name, index, descriptor } => Self::LazyArtifact {
                production: production.clone(),
                name: name.clone(),
                index: *index,
                descriptor: descriptor.clone(),
            },
            Self::LazyArtifactFamily { production, name, descriptor } => Self::LazyArtifactFamily {
                production: production.clone(),
                name: name.clone(),
                descriptor: descriptor.clone(),
            },
            Self::StagedArtifact { production, name, index, descriptor } => Self::StagedArtifact {
                production: production.clone(),
                name: name.clone(),
                index: *index,
                descriptor: descriptor.clone(),
            },
            Self::StagedArtifactFamily { production, name, descriptor } => {
                Self::StagedArtifactFamily {
                    production: production.clone(),
                    name: name.clone(),
                    descriptor: descriptor.clone(),
                }
            }
            Self::IndexedFamily(values) => Self::IndexedFamily(values.clone()),
        }
    }
}

impl<B: Backend> RuntimeValue<B> {
    pub(crate) fn releases_backend_resources_on_drop(&self) -> bool {
        match self {
            Self::Matrix(matrix) => Arc::strong_count(matrix) == 1,
            Self::SmallMatrix(matrix) => Arc::strong_count(matrix) == 1,
            Self::Trapdoor { secret, public, .. } => {
                Arc::strong_count(public) == 1 ||
                    secret.as_ref().is_some_and(|secret| Arc::strong_count(secret) == 1)
            }
            Self::IndexedFamily(values) => {
                values.iter().any(Self::releases_backend_resources_on_drop)
            }
            Self::Int(_) |
            Self::Real(_) |
            Self::Bool(_) |
            Self::Bytes(_) |
            Self::TypedBlob(_) |
            Self::LazyArtifact { .. } |
            Self::LazyArtifactFamily { .. } |
            Self::StagedArtifact { .. } |
            Self::StagedArtifactFamily { .. } => false,
        }
    }
}

impl<B: Backend> RuntimeValue<B> {
    pub fn matrix(value: B::Matrix) -> Self {
        Self::Matrix(Arc::new(value))
    }

    pub fn small_matrix(value: B::SmallMatrix) -> Self {
        Self::SmallMatrix(Arc::new(value))
    }
}
