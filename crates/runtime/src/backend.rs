use mxx_ir_core::{
    ParamEnv,
    node::{ConcatAxis, ConstantMatrix, HashVariant, IndexRange, SampleRange},
    types::ConcreteMatrixType,
};
use num_bigint::BigInt;
use std::fmt::Debug;

pub mod poly;
#[cfg(feature = "gpu")]
mod poly_gpu;

#[derive(Clone, Debug)]
pub struct PreimageRequest<M, T> {
    pub matrix_type: ConcreteMatrixType,
    pub sigma: f64,
    pub trapdoor: T,
    pub public: M,
    pub target: M,
}

pub trait Backend {
    type Matrix: Clone + Debug + PartialEq + Send + Sync;
    type Trapdoor: Clone + Debug + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

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
    fn sub(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn multiply(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn negate(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error>;
    fn scale_integer(
        &mut self,
        value: &Self::Matrix,
        scalar: &BigInt,
    ) -> Result<Self::Matrix, Self::Error>;
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
        inputs: &[Self::Matrix],
        axis: ConcatAxis,
    ) -> Result<Self::Matrix, Self::Error>;
    fn reshape(
        &mut self,
        value: &Self::Matrix,
        rows: usize,
        columns: usize,
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
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        variant: HashVariant,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_trapdoor(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
    ) -> Result<(Self::Matrix, Self::Trapdoor), Self::Error>;
    fn sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<Self::Matrix, Self::Trapdoor>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
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
    fn gadget_decompose(
        &mut self,
        value: &Self::Matrix,
        small: bool,
    ) -> Result<Self::Matrix, Self::Error>;
    fn modulus_down(
        &mut self,
        value: &Self::Matrix,
        target_modulus: &BigInt,
    ) -> Result<Self::Matrix, Self::Error>;
    fn modulus_up(
        &mut self,
        value: &Self::Matrix,
        target_type: &ConcreteMatrixType,
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

    fn matrix_to_bytes(&self, value: &Self::Matrix) -> Vec<u8>;
    fn matrix_from_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error>;
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
    Matrix(B::Matrix),
    Trapdoor {
        secret: Option<B::Trapdoor>,
        public: B::Matrix,
        matrix_type: ConcreteMatrixType,
        sigma: f64,
        gadget_small: Option<bool>,
    },
    LazyArtifact {
        production: mxx_ir_core::artifact::ProductionId,
        name: String,
        index: Option<usize>,
        matrix_type: ConcreteMatrixType,
    },
}

impl<B: Backend> Clone for RuntimeValue<B> {
    fn clone(&self) -> Self {
        match self {
            Self::Int(value) => Self::Int(value.clone()),
            Self::Real(value) => Self::Real(*value),
            Self::Bool(value) => Self::Bool(*value),
            Self::Bytes(value) => Self::Bytes(value.clone()),
            Self::Matrix(value) => Self::Matrix(value.clone()),
            Self::Trapdoor { secret, public, matrix_type, sigma, gadget_small } => Self::Trapdoor {
                secret: secret.clone(),
                public: public.clone(),
                matrix_type: matrix_type.clone(),
                sigma: *sigma,
                gadget_small: *gadget_small,
            },
            Self::LazyArtifact { production, name, index, matrix_type } => Self::LazyArtifact {
                production: production.clone(),
                name: name.clone(),
                index: *index,
                matrix_type: matrix_type.clone(),
            },
        }
    }
}
