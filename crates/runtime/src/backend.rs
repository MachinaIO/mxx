use mxx_ir_core::{
    ParamEnv,
    node::{ConcatAxis, ConstantMatrix, HashVariant},
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

/// Host-backed consecutive columns of a typed preimage relation.
///
/// Each payload is an ordinary backend matrix serialization for the recorded
/// column range. Keeping the chunks outside `Backend::Matrix` prevents a
/// sampled `O((log q)^2)` witness from remaining resident on one GPU.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkedPreimage {
    pub matrix_type: ConcreteMatrixType,
    pub chunks: Vec<PreimageColumnChunk>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreimageColumnChunk {
    pub start: usize,
    pub columns: usize,
    pub bytes: Vec<u8>,
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

/// One independent uniform-residue sampling request.
///
/// The executor groups requests with the same concrete type and range before
/// passing them to a backend.  Keeping the request explicit preserves the
/// original order when a backend returns the sampled matrices.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UniformSampleRequest {
    pub matrix_type: ConcreteMatrixType,
    pub range: SampleRange,
}

/// One independent plain hash-sampling request.
///
/// `key` and `tag` are intentionally request data rather than batching keys:
/// a batch must retain the scalar sampler's domain separation for every draw.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashSampleRequest {
    pub matrix_type: ConcreteMatrixType,
    pub key: [u8; 32],
    pub tag: Vec<u8>,
    pub variant: HashVariant,
    pub gadget_layout: Option<(BigInt, usize)>,
}

pub trait Backend {
    type Matrix: Clone + Debug + PartialEq + Send + Sync;
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
    /// Applies a gadget decomposition and multiplication without materializing
    /// the decomposed right-hand matrix.  Backends must implement this using
    /// their native column-chunked primitive.
    fn mul_decompose(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
        small: bool,
    ) -> Result<Self::Matrix, Self::Error>;
    fn mul_decompose_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>, bool)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs
            .into_iter()
            .map(|(left, right, small)| self.mul_decompose(&left, &right, small))
            .collect()
    }
    /// Applies a host-backed typed preimage one consecutive column chunk at a time.
    fn apply_chunked_preimage(
        &mut self,
        left: &Self::Matrix,
        preimage: &ChunkedPreimage,
    ) -> Result<Self::Matrix, Self::Error> {
        let mut products = Vec::with_capacity(preimage.chunks.len());
        for chunk in &preimage.chunks {
            let mut chunk_type = preimage.matrix_type.clone();
            chunk_type.columns = chunk.columns;
            let right = self.matrix_from_bytes(&chunk_type, &chunk.bytes)?;
            products.push(self.multiply(left, &right)?);
        }
        let references = products.iter().collect::<Vec<_>>();
        self.concat(&references, ConcatAxis::Columns)
    }
    fn apply_chunked_preimage_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, Arc<ChunkedPreimage>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs
            .into_iter()
            .map(|(left, preimage)| self.apply_chunked_preimage(&left, &preimage))
            .collect()
    }
    fn materialize_chunked_preimage(
        &mut self,
        preimage: &ChunkedPreimage,
    ) -> Result<Self::Matrix, Self::Error> {
        let mut matrices = Vec::with_capacity(preimage.chunks.len());
        for chunk in &preimage.chunks {
            let mut chunk_type = preimage.matrix_type.clone();
            chunk_type.columns = chunk.columns;
            matrices.push(self.matrix_from_bytes(&chunk_type, &chunk.bytes)?);
        }
        let references = matrices.iter().collect::<Vec<_>>();
        self.concat(&references, ConcatAxis::Columns)
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
    fn ring_automorphism(
        &mut self,
        value: &Self::Matrix,
        index: usize,
    ) -> Result<Self::Matrix, Self::Error>;
    fn ring_automorphism_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, usize)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(value, index)| self.ring_automorphism(&value, index)).collect()
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
    /// Writes an ordered wave of one-column matrices into an existing
    /// row-major destination. Implementations may override this sink to keep
    /// the final matrix allocation stable while loop waves are released.
    fn write_columns(
        &mut self,
        target: &mut Self::Matrix,
        offset: usize,
        columns: &[Self::Matrix],
    ) -> Result<(), Self::Error>;
    fn sample_uniform(
        &mut self,
        ty: &ConcreteMatrixType,
        range: &SampleRange,
    ) -> Result<Self::Matrix, Self::Error>;
    /// Samples independent uniform requests in input order.
    ///
    /// The default is deliberately scalar so every backend remains correct
    /// while a device backend can override this hook with one batched launch.
    fn sample_uniform_batch(
        &mut self,
        requests: Vec<UniformSampleRequest>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
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
    ) -> Result<Self::Matrix, Self::Error>;
    fn sample_hash(
        &mut self,
        ty: &ConcreteMatrixType,
        key: [u8; 32],
        tag: &[u8],
        variant: HashVariant,
        gadget_layout: Option<(&BigInt, usize)>,
    ) -> Result<Self::Matrix, Self::Error>;
    /// Samples independent hash requests in input order.
    ///
    /// Backends may batch only requests with a compatible concrete layout;
    /// callers already partition by placement and static node shape.
    fn sample_hash_batch(
        &mut self,
        requests: Vec<HashSampleRequest>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
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
    ) -> Result<Self::Matrix, Self::Error>;
    /// Samples and immediately offloads consecutive preimage column chunks.
    fn sample_chunked_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &Self::Matrix,
        column_chunk_width: usize,
    ) -> Result<ChunkedPreimage, Self::Error> {
        let request = PreimageRequest {
            matrix_type: ty.clone(),
            sigma,
            gadget_base: gadget_base.clone(),
            digit_count,
            max_coefficient_bound: max_coefficient_bound.clone(),
            trapdoor: Arc::new(trapdoor.clone()),
            public: Arc::new(public.clone()),
            target: Arc::new(target.clone()),
        };
        let mut batches = self.sample_chunked_preimage_batches_by_placement(
            vec![(self.active_placement(), vec![request])],
            column_chunk_width,
        )?;
        Ok(batches
            .pop()
            .expect("one chunked preimage placement")
            .1
            .pop()
            .expect("one chunked preimage request"))
    }
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
    ) -> Result<Vec<(usize, Vec<Self::Matrix>)>, Self::Error> {
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
    fn sample_chunked_preimage_batches_by_placement(
        &mut self,
        batches: Vec<(usize, Vec<PreimageRequest<Self::Matrix, Self::Trapdoor>>)>,
        column_chunk_width: usize,
    ) -> Result<Vec<(usize, Vec<ChunkedPreimage>)>, Self::Error> {
        assert!(column_chunk_width > 0, "preimage column chunk width must be nonzero");
        let original = self.active_placement();
        let result = (|| {
            let mut all_outputs = Vec::with_capacity(batches.len());
            for (placement, requests) in batches {
                assert!(self.set_active_placement(placement), "backend rejected its own placement");
                let Some(first) = requests.first() else {
                    all_outputs.push((placement, Vec::new()));
                    continue;
                };
                let total_columns = first.matrix_type.columns;
                let ranges = (0..total_columns)
                    .step_by(column_chunk_width)
                    .map(|start| {
                        let end = (start + column_chunk_width).min(total_columns);
                        (start, end)
                    })
                    .collect::<Vec<_>>();
                // Offload target columns before sampling. Every sampling wave below restores
                // only its selected range, so the sampler never receives the full target.
                let mut staged_targets = Vec::with_capacity(requests.len());
                for request in &requests {
                    let mut chunks = Vec::with_capacity(ranges.len());
                    for &(start, end) in &ranges {
                        let target = self.slice(
                            request.target.as_ref(),
                            None,
                            Some(&IndexRange { start, end }),
                        )?;
                        chunks.push(self.matrix_to_bytes(&target));
                    }
                    staged_targets.push(chunks);
                }
                let mut outputs = requests
                    .iter()
                    .map(|request| ChunkedPreimage {
                        matrix_type: request.matrix_type.clone(),
                        chunks: Vec::with_capacity(total_columns.div_ceil(column_chunk_width)),
                    })
                    .collect::<Vec<_>>();
                for (chunk_index, &(start, end)) in ranges.iter().enumerate() {
                    let columns = end - start;
                    tracing::debug!(
                        placement,
                        chunk_index,
                        start,
                        end,
                        columns,
                        request_count = requests.len(),
                        column_chunk_width,
                        "preimage sampling column wave"
                    );
                    let mut chunk_requests = Vec::with_capacity(requests.len());
                    for (request, target_chunks) in requests.iter().zip(&staged_targets) {
                        assert_eq!(
                            request.matrix_type.columns, total_columns,
                            "chunked preimage batch requires one column count"
                        );
                        let mut matrix_type = request.matrix_type.clone();
                        matrix_type.columns = columns;
                        let target =
                            self.matrix_from_bytes(&matrix_type, &target_chunks[chunk_index])?;
                        chunk_requests.push(PreimageRequest {
                            matrix_type,
                            sigma: request.sigma,
                            gadget_base: request.gadget_base.clone(),
                            digit_count: request.digit_count,
                            max_coefficient_bound: request.max_coefficient_bound.clone(),
                            trapdoor: request.trapdoor.clone(),
                            public: request.public.clone(),
                            target: Arc::new(target),
                        });
                    }
                    let sampled = self.sample_preimage_batch(chunk_requests)?;
                    for (output, matrix) in outputs.iter_mut().zip(sampled) {
                        output.chunks.push(PreimageColumnChunk {
                            start,
                            columns,
                            bytes: self.matrix_to_bytes(&matrix),
                        });
                    }
                }
                all_outputs.push((placement, outputs));
            }
            Ok(all_outputs)
        })();
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
    /// A deferred GadgetDecompose result.  The original matrix remains live
    /// so an adjacent ApplyPreimage can use the backend's fused operation.
    DeferredGadgetDecomposition {
        source: Arc<B::Matrix>,
        small: bool,
    },
    ChunkedPreimage(Arc<ChunkedPreimage>),
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
    Family(Vec<RuntimeValue<B>>),
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
            Self::DeferredGadgetDecomposition { source, small } => {
                Self::DeferredGadgetDecomposition { source: source.clone(), small: *small }
            }
            Self::ChunkedPreimage(value) => Self::ChunkedPreimage(value.clone()),
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
            Self::Family(values) => Self::Family(values.clone()),
        }
    }
}

impl<B: Backend> RuntimeValue<B> {
    pub(crate) fn releases_backend_resources_on_drop(&self) -> bool {
        match self {
            Self::Matrix(matrix) => Arc::strong_count(matrix) == 1,
            Self::DeferredGadgetDecomposition { source, .. } => Arc::strong_count(source) == 1,
            Self::ChunkedPreimage(_) => false,
            Self::Trapdoor { secret, public, .. } => {
                Arc::strong_count(public) == 1 ||
                    secret.as_ref().is_some_and(|secret| Arc::strong_count(secret) == 1)
            }
            Self::Family(values) => values.iter().any(Self::releases_backend_resources_on_drop),
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
}
