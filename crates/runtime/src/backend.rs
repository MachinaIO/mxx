use crate::{gpu_execution_plan::FrozenGpuPlan, transcript::DrawSite};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix, MatrixBinaryOp},
    types::{ConcreteMatrixType, InstantiationFrame},
};
use mxx_primitives::matrix::{PolyMatrix, PolyMatrixColumnSource};
use num_bigint::BigInt;
use std::{
    collections::BTreeMap,
    fmt::{self, Debug},
    sync::Arc,
};

pub mod poly;
#[cfg(feature = "gpu")]
pub mod poly_gpu;

#[derive(Clone, Debug)]
pub struct PreimageRequest<M, T> {
    pub instance_slot: usize,
    /// Fixed-plan site/layout contract for the sampled instance.  Legacy
    /// callers leave this absent; fixed executor paths always populate it.
    pub fixed_metadata: Option<PlannedNodeBatchRequest>,
    pub matrix_type: ConcreteMatrixType,
    pub sigma: f64,
    pub gadget_base: BigInt,
    pub digit_count: usize,
    pub max_coefficient_bound: BigInt,
    pub trapdoor: Arc<T>,
    pub public: Arc<M>,
    pub target: Arc<dyn PolyMatrixColumnSource<M>>,
    /// Seed for deterministic GPU sampling with a fixed column schedule.
    pub randomness_seed: [u8; 32],
}

/// Metadata passed to a backend immediately before a fixed node batch is
/// submitted.  It deliberately contains no matrix owner or native pointer;
/// those are bound by the backend to the current runtime values.  The
/// executor keeps the original instance slots so a backend cannot accidentally
/// use a compressed batch index as a sampling or output coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedNodeBatchRequest {
    pub site: u64,
    pub shape_class: u64,
    pub instance_class: u64,
    pub operation_identity: [u8; 32],
    pub implementation_variant: String,
    pub output_layouts: Vec<crate::gpu_execution_plan::LayoutId>,
    /// Explicit port-to-layout bindings. `output_layouts` is retained as the
    /// compact plan representation, while this field prevents a backend from
    /// accidentally treating a reordered fused output as positional-only.
    pub output_port_layouts: Vec<(usize, crate::gpu_execution_plan::LayoutId)>,
    /// Concrete source layouts checked at submit time.  `layout_id` is absent
    /// when a source is a runtime value rather than a planned output.
    pub source_layouts: Vec<PlannedLayoutMetadata>,
    pub output_layout_metadata: Vec<PlannedLayoutMetadata>,
    pub columns_per_job: Vec<usize>,
    pub instance_slots: Vec<usize>,
    pub instance_paths: Vec<Vec<InstantiationFrame>>,
    pub draw_sites: Vec<Option<DrawSite>>,
    pub randomness_seeds: Vec<Option<[u8; 32]>>,
    pub output_ports: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedLayoutMetadata {
    pub layout_id: Option<crate::gpu_execution_plan::LayoutId>,
    pub rows: usize,
    pub columns: usize,
    pub ring_dimension: usize,
    pub representation: String,
}

/// Concrete storage facts supplied by a backend to the warmup planner.  This
/// is intentionally keyed by the concrete matrix type rather than by a
/// caller-provided tower count: the backend is the authority for the ordered
/// CRT basis, active level, representation, and native limb width.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct BackendStorageDescriptor {
    pub representation: String,
    pub ordered_crt_basis: Vec<u64>,
    pub level: usize,
    pub limb_bytes: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendStorageContract {
    pub descriptors: BTreeMap<ConcreteMatrixType, BackendStorageDescriptor>,
    pub active_crt_towers: usize,
    pub crt_limb_bytes: usize,
}

/// Validate the structural part of a backend storage contract before a
/// backend-specific equality check.  In particular, a nonempty set of
/// concrete matrix types may not be represented by an empty descriptor map or
/// by a fabricated one-tower summary of a deeper CRT basis.
pub fn validate_backend_storage_contract(
    types: &[ConcreteMatrixType],
    contract: &BackendStorageContract,
) -> Result<(), String> {
    if contract.active_crt_towers == 0 || contract.crt_limb_bytes == 0 {
        return Err("GPU storage contract has no active CRT towers or limb width".into());
    }
    let mut expected = types.to_vec();
    expected.sort_unstable();
    expected.dedup();
    let actual = contract.descriptors.keys().cloned().collect::<Vec<_>>();
    if actual != expected {
        return Err("GPU storage contract descriptor types do not match concrete inputs".into());
    }
    let deepest = contract
        .descriptors
        .values()
        .map(|descriptor| descriptor.level.saturating_add(1))
        .max()
        .unwrap_or(0);
    if deepest != contract.active_crt_towers {
        return Err("GPU storage contract active CRT depth disagrees with its descriptors".into());
    }
    if contract.descriptors.values().any(|descriptor| {
        descriptor.representation.is_empty() ||
            descriptor.ordered_crt_basis.is_empty() ||
            descriptor.level >= descriptor.ordered_crt_basis.len() ||
            descriptor.limb_bytes != contract.crt_limb_bytes
    }) {
        return Err("GPU storage contract contains an invalid physical descriptor".into());
    }
    Ok(())
}

/// Existing fused primitives grouped across sibling instances. These are
/// short-lived operands, never part of the frozen metadata plan.
pub enum FusedBatchRequest<M, S> {
    RowSum {
        metadata: Option<PlannedNodeBatchRequest>,
        source: Arc<M>,
        right: Option<Arc<M>>,
        rows: Vec<Vec<usize>>,
    },
    Decompose {
        metadata: Option<PlannedNodeBatchRequest>,
        blocks: Vec<Arc<M>>,
        small: bool,
        digits: usize,
    },
    SmallProduct {
        metadata: Option<PlannedNodeBatchRequest>,
        blocks: Vec<Arc<M>>,
        rhs: Arc<S>,
    },
    Add {
        metadata: Option<PlannedNodeBatchRequest>,
        blocks: Vec<Arc<M>>,
        right: Arc<M>,
    },
}

pub enum FusedBatchOutput<M, S> {
    Matrices(Vec<M>),
    Small(S),
}

/// A sibling batch for ordinary fixed-plan operations.  The metadata is
/// carried per original instance rather than per compressed batch position;
/// a fleet backend can therefore rotate owners using the frozen slot/path
/// without rediscovering a pilot or selecting a different implementation.
pub enum FixedOperationBatchRequest<M, S> {
    GeneratedConstant {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        value: ConstantMatrix,
        env: ParamEnv,
    },
    MatrixBinary {
        metadata: PlannedNodeBatchRequest,
        operation: MatrixBinaryOp,
        left: Arc<M>,
        right: Arc<M>,
    },
    MatrixMulSmallRhs {
        metadata: PlannedNodeBatchRequest,
        left: Arc<M>,
        right: Arc<S>,
    },
    MatrixMulAccumulate {
        metadata: PlannedNodeBatchRequest,
        request: MatrixMulAccumulateRequest<M>,
    },
    Negate {
        metadata: PlannedNodeBatchRequest,
        value: Arc<M>,
    },
    Scale {
        metadata: PlannedNodeBatchRequest,
        value: Arc<M>,
        scalar: BigInt,
    },
    UnaryTransform {
        metadata: PlannedNodeBatchRequest,
        operation: FixedUnaryOperation,
        value: Arc<M>,
    },
    Tensor {
        metadata: PlannedNodeBatchRequest,
        left: Arc<M>,
        right: Arc<M>,
    },
    Concat {
        metadata: PlannedNodeBatchRequest,
        inputs: Vec<Arc<M>>,
        axis: ConcatAxis,
    },
    CrtRecompose {
        metadata: PlannedNodeBatchRequest,
        levels: Vec<M>,
        plaintext_moduli: Vec<BigInt>,
        reconstruction_coefficients: Vec<BigInt>,
        destination: ConcreteMatrixType,
    },
}

#[derive(Clone, Debug)]
pub enum FixedUnaryOperation {
    RingAutomorphism {
        index: usize,
    },
    ModulusSwitch {
        destination: ConcreteMatrixType,
    },
    ReduceModulus {
        destination: ConcreteMatrixType,
    },
    CenteredRebase {
        destination: ConcreteMatrixType,
    },
    RnsModUp {
        destination: ConcreteMatrixType,
        source_moduli: Vec<u64>,
        digit_size: usize,
        normalize: bool,
    },
    RnsModDown {
        destination: ConcreteMatrixType,
        source_moduli: Vec<u64>,
        plaintext_modulus: u64,
    },
    Transpose,
    Slice {
        rows: Option<IndexRange>,
        columns: Option<IndexRange>,
    },
}

/// Fixed-plan generated-column requests.  Sampling backends receive one
/// request per original instance, so owner rotation never depends on a
/// compressed batch ordinal.
pub enum FixedGenerationRequest {
    Uniform {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        range: SampleRange,
    },
    Gaussian {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: BigInt,
    },
    Hash {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        key: [u8; 32],
        tag: Vec<u8>,
    },
    HashDecomposed {
        metadata: PlannedNodeBatchRequest,
        ty: ConcreteMatrixType,
        key: [u8; 32],
        tag: Vec<u8>,
        gadget_base: BigInt,
        digit_count: usize,
        small: bool,
    },
}

pub enum FixedGenerationOutput<M, S> {
    Matrix(M),
    Small(S),
}

pub struct FixedGadgetDecomposeRequest<M> {
    pub metadata: PlannedNodeBatchRequest,
    pub input: Arc<M>,
    pub small: bool,
    pub digits: usize,
}

/// Full logical preimage target whose expanded columns are loaded on demand.
/// The staged constructor owns host bytes, allowing a GPU owner to be dropped
/// before sampling while preserving the original matrix dimensions.
pub struct PreimageTarget<M> {
    rows: usize,
    columns: usize,
    loader: Arc<dyn Fn(usize, usize) -> M + Send + Sync>,
}

impl<M> Clone for PreimageTarget<M> {
    fn clone(&self) -> Self {
        Self { rows: self.rows, columns: self.columns, loader: self.loader.clone() }
    }
}

impl<M> Debug for PreimageTarget<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreimageTarget")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl<M> PreimageTarget<M>
where
    M: PolyMatrix + 'static,
{
    pub fn staged(
        params: <M::P as mxx_primitives::poly::Poly>::Params,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Self {
        let loader = Arc::new(move |start: usize, end: usize| {
            M::from_cpu_staging_columns(&params, bytes.as_slice(), start, end)
        });
        Self { rows, columns, loader }
    }

    pub fn resident(value: Arc<M>) -> Self {
        let rows = value.row_size();
        let columns = value.col_size();
        let loader = Arc::new(move |start: usize, end: usize| value.slice_columns(start, end));
        Self { rows, columns, loader }
    }
}

impl<M> PolyMatrixColumnSource<M> for PreimageTarget<M>
where
    M: PolyMatrix + 'static,
{
    fn row_size(&self) -> usize {
        self.rows
    }

    fn col_size(&self) -> usize {
        self.columns
    }

    fn load_columns(&self, start: usize, end: usize) -> M {
        assert!(start <= end && end <= self.columns, "invalid preimage target column interval");
        (self.loader)(start, end)
    }
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
    type Matrix: Clone + Debug + PartialEq + Send + Sync + 'static;
    type SmallMatrix: Clone + Debug + PartialEq + Send + Sync;
    type Trapdoor: Clone + Debug + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    fn fused_batch(
        &mut self,
        requests: Vec<FusedBatchRequest<Self::Matrix, Self::SmallMatrix>>,
    ) -> Result<Vec<FusedBatchOutput<Self::Matrix, Self::SmallMatrix>>, Self::Error> {
        requests
            .into_iter()
            .map(|request| match request {
                FusedBatchRequest::RowSum { source, right, rows, metadata: _ } => {
                    let output = match right {
                        Some(right) => self.tensor_sum_rows(&source, &right, &rows),
                        None => self.sum_rows(&source, &rows),
                    }?;
                    Ok(FusedBatchOutput::Matrices(vec![output]))
                }
                FusedBatchRequest::Decompose { blocks, small, digits, metadata: _ } => self
                    .gadget_decompose_row_blocks(
                        &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                        small,
                        Some(digits),
                    )
                    .map(FusedBatchOutput::Small),
                FusedBatchRequest::SmallProduct { blocks, rhs, metadata: _ } => self
                    .multiply_small_rhs_row_blocks(
                        &blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(),
                        &rhs,
                    )
                    .map(FusedBatchOutput::Matrices),
                FusedBatchRequest::Add { blocks, right, metadata: _ } => self
                    .add_row_blocks(&blocks.iter().map(Arc::as_ref).collect::<Vec<_>>(), &right)
                    .map(|value| FusedBatchOutput::Matrices(vec![value])),
            })
            .collect()
    }

    /// Dispatches one fixed-plan sibling batch.  The default implementation
    /// preserves CPU semantics, while GPU fleets override this hook to submit
    /// the complete wave as one owner-rotated batch.  In particular, callers
    /// must not replace this with `select_gpu_operation` or a pilot-derived
    /// single-instance dispatch when a frozen plan is active.
    fn fixed_operation_batch(
        &mut self,
        requests: Vec<FixedOperationBatchRequest<Self::Matrix, Self::SmallMatrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        requests
            .into_iter()
            .map(|request| match request {
                FixedOperationBatchRequest::GeneratedConstant { ty, value, env, metadata: _ } => {
                    self.constant_matrix(&ty, &value, &env)
                }
                FixedOperationBatchRequest::MatrixBinary {
                    operation,
                    left,
                    right,
                    metadata: _,
                } => match operation {
                    MatrixBinaryOp::Add => self.add(&left, &right),
                    MatrixBinaryOp::Subtract => self.sub(&left, &right),
                    MatrixBinaryOp::Multiply => self.multiply(&left, &right),
                },
                FixedOperationBatchRequest::MatrixMulSmallRhs { left, right, metadata: _ } => {
                    self.multiply_small_rhs(&left, &right)
                }
                FixedOperationBatchRequest::MatrixMulAccumulate { request, metadata: _ } => {
                    self.matrix_mul_accumulate(request)
                }
                FixedOperationBatchRequest::Negate { value, metadata: _ } => self.negate(&value),
                FixedOperationBatchRequest::Scale { value, scalar, metadata: _ } => {
                    self.scale_integer(&value, &scalar)
                }
                FixedOperationBatchRequest::UnaryTransform { operation, value, metadata: _ } => {
                    match operation {
                        FixedUnaryOperation::RingAutomorphism { index } => {
                            self.ring_automorphism(&value, index)
                        }
                        FixedUnaryOperation::ModulusSwitch { destination } => {
                            self.modulus_switch(&value, &destination)
                        }
                        FixedUnaryOperation::ReduceModulus { destination } => {
                            self.reduce_modulus(&value, &destination)
                        }
                        FixedUnaryOperation::CenteredRebase { destination } => {
                            self.centered_rebase(&value, &destination)
                        }
                        FixedUnaryOperation::RnsModUp {
                            destination,
                            source_moduli,
                            digit_size,
                            normalize,
                        } => self.rns_mod_up(
                            &value,
                            &destination,
                            &source_moduli,
                            digit_size,
                            normalize,
                        ),
                        FixedUnaryOperation::RnsModDown {
                            destination,
                            source_moduli,
                            plaintext_modulus,
                        } => self.rns_mod_down(
                            &value,
                            &destination,
                            &source_moduli,
                            plaintext_modulus,
                        ),
                        FixedUnaryOperation::Transpose => self.transpose(&value),
                        FixedUnaryOperation::Slice { rows, columns } => {
                            self.slice(&value, rows.as_ref(), columns.as_ref())
                        }
                    }
                }
                FixedOperationBatchRequest::Tensor { left, right, metadata: _ } => {
                    self.tensor(&left, &right)
                }
                FixedOperationBatchRequest::Concat { inputs, axis, metadata: _ } => {
                    let refs = inputs.iter().map(Arc::as_ref).collect::<Vec<_>>();
                    self.concat(&refs, axis)
                }
                FixedOperationBatchRequest::CrtRecompose {
                    levels,
                    plaintext_moduli,
                    reconstruction_coefficients,
                    destination,
                    metadata: _,
                } => self.crt_recompose(
                    &levels,
                    &plaintext_moduli,
                    &reconstruction_coefficients,
                    &destination,
                ),
            })
            .collect()
    }

    fn fixed_generation_batch(
        &mut self,
        requests: Vec<FixedGenerationRequest>,
    ) -> Result<Vec<FixedGenerationOutput<Self::Matrix, Self::SmallMatrix>>, Self::Error> {
        requests
            .into_iter()
            .map(|request| match request {
                FixedGenerationRequest::Uniform { ty, range, metadata: _ } => {
                    self.sample_uniform(&ty, &range).map(FixedGenerationOutput::Matrix)
                }
                FixedGenerationRequest::Gaussian {
                    ty,
                    sigma,
                    max_coefficient_bound,
                    metadata: _,
                } => self
                    .sample_gaussian(&ty, sigma, &max_coefficient_bound)
                    .map(FixedGenerationOutput::Matrix),
                FixedGenerationRequest::Hash { ty, key, tag, metadata: _ } => {
                    self.sample_hash(&ty, key, &tag).map(FixedGenerationOutput::Matrix)
                }
                FixedGenerationRequest::HashDecomposed {
                    ty,
                    key,
                    tag,
                    gadget_base,
                    digit_count,
                    small,
                    metadata: _,
                } => {
                    let output = if small {
                        self.sample_hash_small_decomposed(
                            &ty,
                            key,
                            &tag,
                            &gadget_base,
                            digit_count,
                        )?
                    } else {
                        self.sample_hash_decomposed(&ty, key, &tag, &gadget_base, digit_count)?
                    };
                    Ok(FixedGenerationOutput::Small(output))
                }
            })
            .collect()
    }

    fn fixed_gadget_decompose_batch(
        &mut self,
        requests: Vec<FixedGadgetDecomposeRequest<Self::Matrix>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
        requests
            .into_iter()
            .map(|request| {
                self.gadget_decompose(&request.input, request.small, Some(request.digits))
            })
            .collect()
    }

    /// Imports one polynomial from coefficient or native evaluation values.
    fn polynomial_from_values(
        &mut self,
        ty: &ConcreteMatrixType,
        values: &[BigInt],
        evaluation: bool,
    ) -> Result<Self::Matrix, Self::Error>;

    /// Exports one scalar polynomial in coefficient or native evaluation order.
    fn polynomial_values(
        &mut self,
        value: &Self::Matrix,
        evaluation: bool,
    ) -> Result<Vec<BigInt>, Self::Error>;

    /// Selects the setup-time GPU calibration for the next primitive. CPU and
    /// non-fleet backends ignore this hook.
    fn select_gpu_operation(&mut self, _operation: [u8; 32]) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Validates backend-specific parts of a frozen plan before the first
    /// dispatch.  CPU backends intentionally accept the metadata and retain
    /// their existing execution semantics.  GPU fleet backends may compare
    /// device identity, representation, and revision here.
    fn validate_frozen_gpu_plan(&self, _plan: &FrozenGpuPlan) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Returns the backend's current runtime contract when it can describe
    /// one. The executor compares it with the frozen contract before any
    /// input materialization or submit, covering backend identity/revision,
    /// logical-to-physical mapping, budgets, and shape/representation/source
    /// layout metadata. Backends without a fleet contract return `None`.
    fn gpu_runtime_contract(
        &self,
        _validated: &mxx_ir_core::ValidatedGraph,
        _inputs: &BTreeMap<String, RuntimeValue<Self>>,
    ) -> Result<Option<crate::gpu_execution_plan::GpuPlanContract>, Self::Error>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Returns backend-authoritative physical storage facts for the concrete
    /// matrix types used by warmup.  GPU implementations must derive these
    /// from their registered native parameters; callers must not synthesize a
    /// tower count or leave the descriptor map empty.
    fn gpu_physical_storage_contract(
        &self,
        _types: &[ConcreteMatrixType],
    ) -> Result<Option<BackendStorageContract>, Self::Error>
    where
        Self: Sized,
    {
        Ok(None)
    }

    /// Compares caller-provided warmup storage metadata with the backend's
    /// native descriptor.  The default is intentionally a no-op for CPU
    /// backends; fleet backends override it to reject stale or fabricated
    /// `active_crt_towers` and descriptor maps before planning.
    fn validate_gpu_physical_storage_contract(
        &self,
        types: &[ConcreteMatrixType],
        supplied: &BackendStorageContract,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        let _ = (types, supplied);
        Ok(())
    }

    /// Freeze public resource limits before warmup. CPU backends have no GPU
    /// contract; fleet backends validate device limits against their contexts.
    fn configure_gpu_plan_budgets(
        &mut self,
        _budgets: &[crate::gpu_execution_plan::GpuDeviceBudget],
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Installs the value-only plan into a backend's fixed-dispatch state.
    /// The default CPU implementation has no state to install.  GPU fleet
    /// implementations should bind the plan without allocating or profiling.
    fn install_frozen_gpu_plan(&mut self, _plan: &FrozenGpuPlan) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Receives the fixed node choice before any primitive is submitted.
    /// Implementations must not use this hook to discover a different width,
    /// owner, device, or pilot result.  The default is the CPU boundary and
    /// leaves the ordinary operation methods untouched.
    fn prepare_fixed_node_batch(
        &mut self,
        _request: &PlannedNodeBatchRequest,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Whether the backend has installed a fixed production plan. This is
    /// separate from `placement_count`: a fleet may expose one executor
    /// placement while dispatching fixed jobs across several logical GPUs.
    fn fixed_plan_active(&self) -> bool {
        false
    }

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
    fn add_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        let left = self.concat(blocks, ConcatAxis::Rows)?;
        self.add(&left, right)
    }
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
    fn ring_automorphism(
        &mut self,
        value: &Self::Matrix,
        index: usize,
    ) -> Result<Self::Matrix, Self::Error>;

    fn modulus_switch(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;
    fn reduce_modulus(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;

    fn centered_rebase(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;

    fn rns_mod_up(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        digit_size: usize,
        normalize: bool,
    ) -> Result<Self::Matrix, Self::Error>;

    fn rns_mod_down(
        &mut self,
        value: &Self::Matrix,
        destination: &ConcreteMatrixType,
        source_moduli: &[u64],
        plaintext_modulus: u64,
    ) -> Result<Self::Matrix, Self::Error>;

    fn ring_automorphism_batch(
        &mut self,
        inputs: Vec<(Arc<Self::Matrix>, usize)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        inputs.into_iter().map(|(value, index)| self.ring_automorphism(&value, index)).collect()
    }

    fn preimage_target(
        &mut self,
        value: Arc<Self::Matrix>,
    ) -> Result<(Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Arc<Vec<u8>>), Self::Error>;

    fn matrix_from_cpu_staging_bytes(
        &self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error>;

    fn preimage_target_from_staging(
        &self,
        ty: &ConcreteMatrixType,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Result<Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Self::Error>;

    fn validate_preimage_bound(
        &self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
    ) -> Result<(), Self::Error>;

    fn transpose(&mut self, value: &Self::Matrix) -> Result<Self::Matrix, Self::Error>;
    fn slice(
        &mut self,
        value: &Self::Matrix,
        rows: Option<&IndexRange>,
        columns: Option<&IndexRange>,
    ) -> Result<Self::Matrix, Self::Error>;
    /// Sums each nonempty group of source rows into one output row.
    /// Indices must be in bounds; repeated indices retain their multiplicity.
    fn sum_rows(
        &mut self,
        value: &Self::Matrix,
        rows: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        if rows.is_empty() {
            return self.slice(value, Some(&IndexRange { start: 0, end: 0 }), None);
        }
        let output = rows
            .iter()
            .map(|group| {
                let (first, rest) = group.split_first().expect("row sum group must be nonempty");
                let mut sum =
                    self.slice(value, Some(&IndexRange { start: *first, end: first + 1 }), None)?;
                for row in rest {
                    let term =
                        self.slice(value, Some(&IndexRange { start: *row, end: row + 1 }), None)?;
                    sum = self.add(&sum, &term)?;
                }
                Ok(sum)
            })
            .collect::<Result<Vec<_>, Self::Error>>()?;
        self.concat(&output.iter().collect::<Vec<_>>(), ConcatAxis::Rows)
    }
    /// Sum selected tensor-product rows without requiring a materialized tensor.
    fn tensor_sum_rows(
        &mut self,
        left: &Self::Matrix,
        right: &Self::Matrix,
        rows: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        let tensor = self.tensor(left, right)?;
        self.sum_rows(&tensor, rows)
    }
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
        target: &dyn PolyMatrixColumnSource<Self::Matrix>,
        randomness_seed: [u8; 32],
    ) -> Result<Self::SmallMatrix, Self::Error>;

    /// Fixed-plan preimage entry point. GPU fleet implementations consume the
    /// already selected owner intervals and tile widths here. CPU and legacy
    /// backends retain their ordinary sampler semantics through the default.
    fn fixed_sample_preimage(
        &mut self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
        trapdoor: &Self::Trapdoor,
        public: &Self::Matrix,
        target: &dyn PolyMatrixColumnSource<Self::Matrix>,
        randomness_seed: [u8; 32],
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.sample_preimage(
            ty,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
            trapdoor,
            public,
            target,
            randomness_seed,
        )
    }

    fn sample_preimage_batch(
        &mut self,
        requests: Vec<PreimageRequest<Self::Matrix, Self::Trapdoor>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
        requests
            .into_iter()
            .map(|request| {
                let sample = if self.fixed_plan_active() {
                    Self::fixed_sample_preimage
                } else {
                    Self::sample_preimage
                };
                sample(
                    self,
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
    /// Inclusive full-modulus reconstruction error for this backend's regular gadget.
    /// Exact non-CRT backends keep the default zero bound.
    fn gadget_error_bound(
        &self,
        _ty: &ConcreteMatrixType,
        _digit_count: Option<usize>,
    ) -> Result<BigInt, Self::Error> {
        Ok(BigInt::from(0u8))
    }
    fn gadget_decompose(
        &mut self,
        value: &Self::Matrix,
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error>;
    fn multiply_small_rhs(
        &mut self,
        lhs: &Self::Matrix,
        rhs: &Self::SmallMatrix,
    ) -> Result<Self::Matrix, Self::Error>;
    fn gadget_decompose_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        small: bool,
        digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        let value = self.concat(blocks, ConcatAxis::Rows)?;
        self.gadget_decompose(&value, small, digit_count)
    }
    fn multiply_small_rhs_row_blocks(
        &mut self,
        blocks: &[&Self::Matrix],
        rhs: &Self::SmallMatrix,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        blocks.iter().map(|block| self.multiply_small_rhs(block, rhs)).collect()
    }
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
        destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error>;

    fn matrix_to_bytes(&self, value: &Self::Matrix) -> Vec<u8>;
    fn matrices_to_bytes(&self, values: &[&Self::Matrix]) -> Vec<Vec<u8>> {
        values.iter().map(|value| self.matrix_to_bytes(value)).collect()
    }
    /// Decodes an intact payload from this backend's matching matrix codec and type.
    /// Malformed bytes or mismatched metadata violate the caller contract; implementations
    /// may panic. The result reports supported backend failures, not arbitrary corruption.
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

/// A caller-supplied value must match its validated concrete wire type in full, including
/// shape, ring/CRT parameters, representation, and sampler metadata. Trapdoor public and
/// secret material must belong together. This obligation applies recursively to families;
/// execution's value-kind checks do not certify it or inspect resident matrix contents.
pub enum RuntimeValue<B: Backend> {
    Int(BigInt),
    Real(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    TypedBlob(Vec<u8>),
    Matrix(Arc<B::Matrix>),
    /// Host-staged matrix; expanded columns are loaded only when consumed.
    HostMatrix {
        matrix_type: ConcreteMatrixType,
        bytes: Arc<Vec<u8>>,
    },
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
            Self::HostMatrix { matrix_type, bytes } => {
                Self::HostMatrix { matrix_type: matrix_type.clone(), bytes: bytes.clone() }
            }
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
            Self::HostMatrix { .. } |
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

#[cfg(test)]
mod storage_contract_tests {
    use super::*;

    fn matrix_type() -> ConcreteMatrixType {
        ConcreteMatrixType { modulus: BigInt::from(17u8), ring_dimension: 8, rows: 1, columns: 2 }
    }

    fn descriptor(level: usize) -> BackendStorageDescriptor {
        BackendStorageDescriptor {
            representation: "full_dcrt".into(),
            ordered_crt_basis: vec![17, 19, 23, 29],
            level,
            limb_bytes: 8,
        }
    }

    #[test]
    fn storage_contract_rejects_empty_map_and_fabricated_one_tower_depth() {
        let ty = matrix_type();
        let empty = BackendStorageContract {
            descriptors: BTreeMap::new(),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
        };
        assert!(validate_backend_storage_contract(std::slice::from_ref(&ty), &empty).is_err());

        let four_tower = BackendStorageContract {
            descriptors: BTreeMap::from([(ty.clone(), descriptor(3))]),
            active_crt_towers: 1,
            crt_limb_bytes: 8,
        };
        assert!(validate_backend_storage_contract(&[ty], &four_tower).is_err());
    }

    #[test]
    fn storage_contract_accepts_ordered_native_descriptor() {
        let ty = matrix_type();
        let contract = BackendStorageContract {
            descriptors: BTreeMap::from([(ty.clone(), descriptor(3))]),
            active_crt_towers: 4,
            crt_limb_bytes: 8,
        };
        validate_backend_storage_contract(&[ty], &contract).unwrap();
    }
}
