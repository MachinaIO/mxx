//! Fixed-resource execution for the standalone GPU RNS ModDown path.

#[cfg(test)]
#[path = "gpu_prepared_sampler_tests.rs"]
mod gpu_prepared_sampler_tests;

use super::{GpuColumnShard, GpuDcrtBackend, GpuFleetMatrix, GpuFleetSmallMatrix};
use crate::backend::Backend;
use mxx_ir_core::{
    ValidatedGraph,
    node::{MatrixBinaryOp, NodeKind},
    types::{Port, WireRef},
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTPolyMatrix, GpuMatrixModulusConversion, GpuMatrixRangeConstant,
            GpuMatrixSampleDist, GpuPreparedAccumulateCommand, GpuPreparedArithmetic,
            GpuPreparedArithmeticCommand, GpuPreparedArithmeticKind, GpuPreparedCenteredRebase,
            GpuPreparedCompactDecompose, GpuPreparedConstCoeffReadback, GpuPreparedCrtRecompose,
            GpuPreparedGadgetDecompose, GpuPreparedHashSample, GpuPreparedInputCopy,
            GpuPreparedModulusCommand, GpuPreparedModulusConversion, GpuPreparedRange,
            GpuPreparedRequest, GpuPreparedRnsUpload, GpuPreparedSampling, GpuPreparedScalarPack,
            GpuPreparedSchedule, GpuPreparedSlotKind, GpuPreparedSmallRhs, GpuPreparedStorage,
            GpuPreparedThreshold, GpuPreparedTransform, GpuPreparedView,
            GpuPreparedWorkspaceLayout, GpuSmallMatrix, GpuTracedClaim,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{GPU_POLY_FORMAT_COEFF, GPU_POLY_FORMAT_EVAL, GpuDCRTPolyParams},
    },
    sampler::trapdoor::gpu::{
        GpuDCRTTrapdoor, GpuPreparedPreimageSampler, GpuPreparedTrapdoorSampler,
    },
};
use num_traits::ToPrimitive;
use rand::{Rng, SeedableRng};
#[path = "gpu_prepared_scalar.rs"]
mod gpu_prepared_scalar;
use gpu_prepared_scalar::{prepare_scalar_commands, stage_runtime_scalar};
use mxx_primitives::matrix::gpu_dcrt_poly::{
    GpuPreparedScalarBuffer, GpuPreparedScalarMatrixSelect, GpuPreparedScalarOp,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Internal prepared-value boundary. Public `RuntimeValue` remains unchanged;
/// warmup converts it once to these owner-bearing values and replay only
/// rebinds the fixed slots.
#[derive(Clone, Debug)]
pub(crate) enum PreparedRuntimeValue {
    FleetMatrix(Arc<GpuFleetMatrix>),
    FleetSmallMatrix(Arc<GpuFleetSmallMatrix>),
    Trapdoor { secret: Arc<super::GpuFleetTrapdoor>, public: Arc<GpuFleetMatrix> },
    Bytes(Arc<[u8]>),
    Int(num_bigint::BigInt),
    Real(f64),
    Bool(bool),
    Family(Arc<[PreparedRuntimeValue]>),
}

fn prepared_family_leaf<'a>(
    value: &'a PreparedRuntimeValue,
    path: &[usize],
) -> Result<&'a PreparedRuntimeValue, String> {
    let mut value = value;
    for index in path {
        let PreparedRuntimeValue::Family(members) = value else {
            return Err("prepared family member path crosses a non-family value".into());
        };
        value = members
            .get(*index)
            .ok_or_else(|| "prepared family member path is outside its root payload".to_owned())?;
    }
    Ok(value)
}

pub(crate) fn expand_prepared_runtime_inputs(
    program: &super::gpu_prepared_lowering::PreparedProgram,
    inputs: &[PreparedRuntimeValue],
) -> Result<Vec<PreparedRuntimeValue>, String> {
    if inputs.len() != program.inputs.len() {
        return Err("prepared input contract has the wrong number of values".into());
    }
    let mut roots = BTreeMap::new();
    for (index, wire) in program.inputs.iter().copied().enumerate() {
        roots.insert(wire, &inputs[index]);
    }
    program
        .runtime_input_wires
        .iter()
        .map(|wire| {
            if let Some(leaf) = program.input_leaf_bindings.get(wire) {
                let root = roots
                    .get(&leaf.root)
                    .ok_or_else(|| "prepared family root input is unavailable".to_owned())?;
                Ok(prepared_family_leaf(root, &leaf.path)?.clone())
            } else {
                roots
                    .get(wire)
                    .cloned()
                    .cloned()
                    .ok_or_else(|| "prepared root input is unavailable".to_owned())
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparedGpuWorkCounters {
    pub graph_traversals: usize,
    pub graph_hashes: usize,
    pub assignments: usize,
    pub admissions: usize,
    pub native_validations: usize,
    pub reservations: usize,
    pub leases: usize,
    pub project_allocations: usize,
    pub dynamic_events: usize,
    pub dynamic_streams: usize,
    pub measurement_launches: usize,
    pub production_kernels: usize,
    pub cuda_allocations: usize,
    pub provisioning_begins: usize,
    pub provisioning_permits: usize,
    pub provisioning_appends: usize,
    pub generic_fallbacks: usize,
    pub input_name_lookups: usize,
    pub topology_scans: usize,
    pub output_reconstructions: usize,
    pub host_allocations: usize,
}

#[cfg(feature = "gpu-instrumentation")]
static PREPARED_WORK_COUNTERS: [AtomicUsize; 14] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_WORK_GATE: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_INPUT_NAME_LOOKUPS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_TOPOLOGY_SCANS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_OUTPUT_RECONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "gpu-instrumentation")]
static PREPARED_HOST_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

pub fn begin_prepared_gpu_work_gate() {
    #[cfg(feature = "gpu-instrumentation")]
    {
        mxx_primitives::poly::dcrt::gpu::gpu_test_reset_work_counters();
        mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(true);
        PREPARED_WORK_GATE.store(1, Ordering::Release);
    }
}

pub fn end_prepared_gpu_work_gate() {
    #[cfg(feature = "gpu-instrumentation")]
    {
        PREPARED_WORK_GATE.store(0, Ordering::Release);
        mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(false);
    }
}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_forbidden(counter: usize) {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_WORK_COUNTERS[counter].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "gpu-instrumentation")]
fn record_prepared_provisioning(counter: usize) {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_WORK_COUNTERS[10 + counter].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_provisioning_begin() {
    record_prepared_provisioning(0);
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_provisioning_begin() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_provisioning_permit() {
    record_prepared_provisioning(1);
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_provisioning_permit() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_provisioning_append() {
    record_prepared_provisioning(2);
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_provisioning_append() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_generic_fallback() {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_WORK_COUNTERS[13].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_input_name_lookup() {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_INPUT_NAME_LOOKUPS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_input_name_lookup() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_topology_scan(count: usize) {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_TOPOLOGY_SCANS.fetch_add(count, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_topology_scan(_: usize) {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_output_reconstruction() {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_OUTPUT_RECONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_output_reconstruction() {}

#[cfg(feature = "gpu-instrumentation")]
pub(crate) fn record_prepared_host_allocation() {
    if PREPARED_WORK_GATE.load(Ordering::Acquire) != 0 {
        PREPARED_HOST_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_host_allocation() {}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_generic_fallback() {}

#[cfg(not(feature = "gpu-instrumentation"))]
#[inline(always)]
pub(crate) fn record_prepared_forbidden(counter: usize) {
    if counter == usize::MAX {
        return;
    }
}

pub fn reset_prepared_gpu_work_counters() {
    #[cfg(feature = "gpu-instrumentation")]
    {
        for counter in &PREPARED_WORK_COUNTERS {
            counter.store(0, Ordering::Relaxed);
        }
        PREPARED_INPUT_NAME_LOOKUPS.store(0, Ordering::Relaxed);
        PREPARED_TOPOLOGY_SCANS.store(0, Ordering::Relaxed);
        PREPARED_OUTPUT_RECONSTRUCTIONS.store(0, Ordering::Relaxed);
        PREPARED_HOST_ALLOCATIONS.store(0, Ordering::Relaxed);
        mxx_primitives::poly::dcrt::gpu::gpu_test_reset_work_counters();
    }
}

pub fn prepared_gpu_work_counters() -> PreparedGpuWorkCounters {
    #[cfg(feature = "gpu-instrumentation")]
    {
        let value = |index: usize| PREPARED_WORK_COUNTERS[index].load(Ordering::Relaxed);
        let (
            native_events,
            native_streams,
            native_validations,
            cuda_allocations,
            production_kernels,
            measurement_launches,
        ) = mxx_primitives::poly::dcrt::gpu::gpu_test_work_counters();
        PreparedGpuWorkCounters {
            graph_traversals: value(0),
            graph_hashes: value(1),
            assignments: value(2),
            admissions: value(3),
            native_validations: value(4) + native_validations,
            reservations: value(5),
            leases: value(6),
            project_allocations: value(7),
            dynamic_events: value(8) + native_events,
            dynamic_streams: native_streams,
            measurement_launches: value(9) + measurement_launches,
            production_kernels,
            cuda_allocations,
            provisioning_begins: value(10),
            provisioning_permits: value(11),
            provisioning_appends: value(12),
            generic_fallbacks: value(13),
            input_name_lookups: PREPARED_INPUT_NAME_LOOKUPS.load(Ordering::Relaxed),
            topology_scans: PREPARED_TOPOLOGY_SCANS.load(Ordering::Relaxed),
            output_reconstructions: PREPARED_OUTPUT_RECONSTRUCTIONS.load(Ordering::Relaxed),
            host_allocations: PREPARED_HOST_ALLOCATIONS.load(Ordering::Relaxed),
        }
    }
    #[cfg(not(feature = "gpu-instrumentation"))]
    {
        PreparedGpuWorkCounters::default()
    }
}

struct PreparedGpuTarget {
    device: i32,
    start: usize,
    parameters: mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
}

/// One owner-bearing operation in the fixed replay tape.  The plan and every
/// matrix it references live in the variant itself; replay therefore cannot
/// rediscover a kernel, allocate scratch, or select a destination.
//
// Scalar/control lowering is assembled separately from owner-bearing GPU
// commands, so those control variants remain explicit in this tape.
#[allow(dead_code)]
pub enum PreparedOperation {
    Trapdoor {
        command: GpuPreparedTrapdoorSampler,
        rng: rand::rngs::StdRng,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Preimage {
        command: GpuPreparedPreimageSampler,
        rng: rand::rngs::StdRng,
        secret: Arc<GpuDCRTTrapdoor>,
        input: Option<(usize, usize)>,
        output: Arc<GpuSmallMatrix>,
        device: i32,
        start: usize,
    },
    ScalarOp {
        command: Arc<GpuPreparedScalarOp>,
        wire: WireRef,
        kind: mxx_ir_core::types::ConcreteWireType,
        device: i32,
    },
    ScalarUpload {
        command: Arc<GpuPreparedScalarBuffer>,
        input: usize,
        wire: WireRef,
        kind: mxx_ir_core::types::ConcreteWireType,
        device: i32,
    },
    Threshold {
        command: Arc<GpuPreparedThreshold>,
        device: i32,
        node: u32,
        output_bool: bool,
    },
    ScalarPack {
        command: Arc<GpuPreparedScalarPack>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    ScalarMatrixSelect {
        command: Arc<GpuPreparedScalarMatrixSelect>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    InputCopy {
        command: GpuPreparedInputCopy,
        input: Option<usize>,
        source: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Arithmetic {
        command: GpuPreparedArithmeticCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Accumulate {
        command: GpuPreparedAccumulateCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Transform {
        command: GpuPreparedTransform,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Modulus {
        command: GpuPreparedModulusCommand,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Transpose {
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedTranspose>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    ConcatRows {
        commands: Box<[GpuPreparedInputCopy]>,
        sources: Box<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    CenteredRebase {
        command: Arc<GpuPreparedCenteredRebase>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    GadgetDecompose {
        command: Arc<GpuPreparedGadgetDecompose>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Sampling {
        command: Arc<GpuPreparedSampling>,
        seed: mxx_primitives::poly::dcrt::gpu::GpuRngSeed,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    SmallRhs {
        command: Arc<GpuPreparedSmallRhs>,
        input: usize,
        source: Option<Arc<GpuDCRTPolyMatrix>>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    HashSample {
        command: Arc<GpuPreparedHashSample>,
        input: usize,
        operand_inputs: Box<[usize]>,
        tag_prefix: Box<[u8]>,
        tag_scratch: Vec<u8>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    CompactDecompose {
        command: Arc<GpuPreparedCompactDecompose>,
        output: Arc<GpuSmallMatrix>,
        device: i32,
        start: usize,
    },
    Reconstruction {
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsReconstruction>,
        in_flight:
            Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsReconstructionInFlight>,
        values: Arc<Mutex<Box<[num_bigint::BigUint]>>>,
        node: u32,
        device: i32,
        start: usize,
    },
    Readback {
        command: Arc<GpuPreparedConstCoeffReadback>,
        in_flight:
            Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedConstCoeffReadbackInFlight>,
        values: Arc<Mutex<Box<[u64]>>>,
        node: u32,
        device: i32,
        start: usize,
    },
    Upload {
        command: Arc<GpuPreparedRnsUpload>,
        in_flight: Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsUploadInFlight>,
        input: usize,
        mode: PreparedUploadMode,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    CrtRecompose {
        command: Arc<GpuPreparedCrtRecompose>,
        levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Alias {
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    Selection {
        candidates: Box<[PreparedSelectionCandidate]>,
        selected: Arc<AtomicUsize>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
    LoopBody {
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparedUploadMode {
    Bytes,
    Constant,
}

pub(crate) struct PreparedSelectionCandidate {
    command: GpuPreparedInputCopy,
    source: Arc<GpuDCRTPolyMatrix>,
}

pub(crate) struct PreparedCommand {
    operation: PreparedOperation,
    pub(crate) stream: u32,
    pub(crate) wait_events: Box<[u32]>,
    pub(crate) completion_event: u32,
    pub(crate) variant: usize,
    selection_result: Option<usize>,
    schedule: Option<Arc<GpuPreparedSchedule>>,
}

#[derive(Debug)]
enum PreparedCommandError {
    SamplingExhausted { column_start: usize, column_count: usize, attempts: usize },
    Gpu(String),
}

impl PreparedCommandError {
    fn from_preimage(
        error: mxx_primitives::sampler::trapdoor::gpu::GpuPreparedPreimageError,
    ) -> Self {
        match error {
            mxx_primitives::sampler::trapdoor::gpu::GpuPreparedPreimageError::Sampling(
                mxx_primitives::matrix::SmallMatrixError::AttemptExhausted {
                    column_start,
                    column_count,
                    attempts,
                },
            ) => Self::SamplingExhausted { column_start, column_count, attempts },
            error => Self::Gpu(error.to_string()),
        }
    }
}

impl std::fmt::Display for PreparedCommandError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SamplingExhausted { column_start, column_count, attempts } => write!(
                formatter,
                "prepared preimage sampling exhausted at columns {column_start}..{} after {attempts} attempts",
                column_start + column_count
            ),
            Self::Gpu(error) => formatter.write_str(error),
        }
    }
}

#[allow(dead_code)]
impl PreparedCommand {
    fn new(operation: PreparedOperation) -> Self {
        Self {
            operation,
            stream: 0,
            wait_events: Box::new([]),
            completion_event: 0,
            variant: 0,
            selection_result: None,
            schedule: None,
        }
    }

    fn attach_schedule(&mut self, schedule: Arc<GpuPreparedSchedule>) {
        assert!(self.schedule.is_none(), "prepared command schedule already attached");
        self.schedule = Some(schedule);
    }

    fn apply_topology(&mut self, node: &super::gpu_prepared_lowering::PreparedTopologyNode) {
        self.stream = node.stream;
        self.wait_events = node.waits.clone();
        self.completion_event = node.completion;
    }

    fn submit_scheduled<T>(&self, result: Result<T, String>) -> Result<T, String> {
        match (&self.schedule, result) {
            (Some(schedule), Ok(value)) => schedule.end().map(|()| value),
            (_, result) => result,
        }
    }

    fn begin_schedule(&self) -> Result<(), String> {
        if let Some(schedule) = &self.schedule {
            schedule.begin()?;
        }
        Ok(())
    }

    pub fn input_copy(
        command: GpuPreparedInputCopy,
        input: usize,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::InputCopy {
            command,
            input: Some(input),
            source: None,
            output,
            device,
            start,
        })
    }

    pub fn input_copy_from_owner(
        command: GpuPreparedInputCopy,
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::InputCopy {
            command,
            input: None,
            source: Some(source),
            output,
            device,
            start,
        })
    }

    pub fn arithmetic(
        command: GpuPreparedArithmeticCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Arithmetic { command, output, device, start })
    }

    pub fn accumulate(
        command: GpuPreparedAccumulateCommand,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Accumulate { command, output, device, start })
    }

    pub fn transform(
        command: GpuPreparedTransform,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Transform { command, target, device, start })
    }

    pub fn modulus(
        command: GpuPreparedModulusCommand,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Modulus { command, target, device, start })
    }

    pub fn transpose(
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedTranspose>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Transpose { command, output, device, start })
    }

    pub fn concat_rows(
        commands: Box<[GpuPreparedInputCopy]>,
        sources: Box<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::ConcatRows { commands, sources, output, device, start })
    }

    pub fn centered_rebase(
        command: Arc<GpuPreparedCenteredRebase>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::CenteredRebase { command, output, device, start })
    }

    pub fn gadget_decompose(
        command: Arc<GpuPreparedGadgetDecompose>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::GadgetDecompose { command, output, device, start })
    }

    pub fn sampling(
        command: Arc<GpuPreparedSampling>,
        seed: mxx_primitives::poly::dcrt::gpu::GpuRngSeed,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Sampling { command, seed, output, device, start })
    }

    pub fn small_rhs(
        command: Arc<GpuPreparedSmallRhs>,
        input: usize,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::SmallRhs {
            command,
            input,
            source: None,
            output,
            device,
            start,
        })
    }

    pub fn small_rhs_from_owner(
        command: Arc<GpuPreparedSmallRhs>,
        source: Arc<GpuDCRTPolyMatrix>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::SmallRhs {
            command,
            input: 0,
            source: Some(source),
            output,
            device,
            start,
        })
    }

    pub fn hash_sample(
        command: Arc<GpuPreparedHashSample>,
        input: usize,
        operand_inputs: Box<[usize]>,
        tag_prefix: Box<[u8]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::HashSample {
            command,
            input,
            operand_inputs,
            tag_scratch: Vec::with_capacity(tag_prefix.len() + 256),
            tag_prefix,
            output,
            device,
            start,
        })
    }

    pub fn compact_decompose(
        command: Arc<GpuPreparedCompactDecompose>,
        output: Arc<GpuSmallMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::CompactDecompose { command, output, device, start })
    }

    pub fn reconstruction(
        command: Arc<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRnsReconstruction>,
        values: Arc<Mutex<Box<[num_bigint::BigUint]>>>,
        node: u32,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Reconstruction {
            command,
            in_flight: None,
            values,
            node,
            device,
            start,
        })
    }

    pub fn readback(
        command: Arc<GpuPreparedConstCoeffReadback>,
        values: Arc<Mutex<Box<[u64]>>>,
        node: u32,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Readback {
            command,
            in_flight: None,
            values,
            node,
            device,
            start,
        })
    }

    pub fn upload(
        command: Arc<GpuPreparedRnsUpload>,
        input: usize,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Upload {
            command,
            in_flight: None,
            input,
            mode: PreparedUploadMode::Bytes,
            target,
            device,
            start,
        })
    }

    pub fn upload_constant(
        command: Arc<GpuPreparedRnsUpload>,
        input: usize,
        target: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Upload {
            command,
            in_flight: None,
            input,
            mode: PreparedUploadMode::Constant,
            target,
            device,
            start,
        })
    }

    pub fn crt_recompose(
        command: Arc<GpuPreparedCrtRecompose>,
        levels: Arc<[Arc<GpuDCRTPolyMatrix>]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::CrtRecompose { command, levels, output, device, start })
    }

    pub fn alias(output: Arc<GpuDCRTPolyMatrix>, device: i32, start: usize) -> Self {
        Self::new(PreparedOperation::Alias { output, device, start })
    }

    pub fn selection(
        candidates: Box<[PreparedSelectionCandidate]>,
        output: Arc<GpuDCRTPolyMatrix>,
        device: i32,
        start: usize,
    ) -> Self {
        Self::new(PreparedOperation::Selection {
            candidates,
            selected: Arc::new(AtomicUsize::new(0)),
            output,
            device,
            start,
        })
    }

    fn set_selection(&mut self, selected: usize) {
        if let PreparedOperation::Selection { selected: current, candidates, .. } = &self.operation
        {
            current.store(selected.min(candidates.len().saturating_sub(1)), Ordering::Release);
        }
    }

    pub fn loop_body(output: Arc<GpuDCRTPolyMatrix>, device: i32, start: usize) -> Self {
        Self::new(PreparedOperation::LoopBody { output, device, start })
    }

    fn submit_operation(&mut self, inputs: &[Arc<GpuDCRTPolyMatrix>]) -> Result<(), String> {
        debug_assert!(self.stream < 32);
        debug_assert!(self.wait_events.iter().all(|event| *event != self.completion_event));
        match &mut self.operation {
            PreparedOperation::Trapdoor { command, rng, .. } => {
                command.submit(std::array::from_fn(|_| {
                    mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes(rng.random())
                }))
            }
            PreparedOperation::Preimage { command, rng, secret, input: None, .. } => {
                command.submit(secret, rng.random())
            }
            PreparedOperation::Preimage { .. } => {
                Err("prepared preimage requires typed trapdoor input".into())
            }
            PreparedOperation::Threshold { command, .. } => command.submit(),
            PreparedOperation::ScalarOp { command, .. } => command.submit(),
            PreparedOperation::ScalarMatrixSelect { command, .. } => command.submit(),
            PreparedOperation::ScalarUpload { .. } => {
                Err("scalar upload requires runtime input".into())
            }
            PreparedOperation::ScalarPack { command, .. } => command.submit(),
            PreparedOperation::InputCopy { command, input, source, .. } => {
                let source = if let Some(source) = source.as_ref() {
                    source.as_ref()
                } else {
                    inputs
                        .get(input.ok_or_else(|| {
                            "prepared input copy source is unavailable".to_owned()
                        })?)
                        .ok_or_else(|| "prepared input copy source is unavailable".to_owned())?
                };
                command.submit_borrowed(source)
            }
            PreparedOperation::Arithmetic { command, .. } => command.submit(),
            PreparedOperation::Accumulate { command, .. } => command.submit(),
            PreparedOperation::Transform { command, target, .. } => command.submit_shared(target),
            PreparedOperation::Modulus { command, target, .. } => command.submit(target),
            PreparedOperation::Transpose { command, .. } => command.submit(),
            PreparedOperation::ConcatRows { commands, sources, .. } => {
                for (command, source) in commands.iter().zip(sources.iter()) {
                    command.submit(Arc::clone(source)).map(|_| ())?;
                }
                Ok(())
            }
            PreparedOperation::CenteredRebase { command, .. } => command.submit(),
            PreparedOperation::GadgetDecompose { command, .. } => command.submit(),
            PreparedOperation::Sampling { command, seed, .. } => command.submit(*seed).map(|_| ()),
            PreparedOperation::SmallRhs { command, input, source, .. } => {
                if let Some(source) = source {
                    command.submit(Arc::clone(source)).map(|_| ())
                } else {
                    let source = inputs
                        .get(*input)
                        .ok_or_else(|| "prepared compact RHS source is unavailable".to_owned())?;
                    command.submit(Arc::clone(source)).map(|_| ())
                }
            }
            PreparedOperation::HashSample { .. } => {
                Err("prepared hash sample requires typed runtime bytes".into())
            }
            PreparedOperation::CompactDecompose { command, .. } => command.submit(),
            PreparedOperation::Reconstruction { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Readback { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Upload { .. } => {
                Err("prepared RNS upload requires typed runtime bytes".into())
            }
            PreparedOperation::CrtRecompose { command, levels, .. } => {
                command.submit(Arc::clone(levels)).map(|_| ())
            }
            PreparedOperation::Alias { .. } | PreparedOperation::LoopBody { .. } => Ok(()),
            PreparedOperation::Selection { candidates, selected, .. } => {
                let index = selected.load(Ordering::Acquire);
                let candidate = candidates
                    .get(index)
                    .ok_or_else(|| "prepared selection candidate is unavailable".to_owned())?;
                candidate.command.submit_borrowed(&candidate.source)
            }
        }
    }

    fn submit(&mut self, inputs: &[Arc<GpuDCRTPolyMatrix>]) -> Result<(), String> {
        self.begin_schedule()?;
        let result = self.submit_operation(inputs);
        self.submit_scheduled(result)
    }

    fn submit_borrowed_operation(&mut self, inputs: &[&GpuDCRTPolyMatrix]) -> Result<(), String> {
        debug_assert!(self.stream < 32);
        debug_assert!(self.wait_events.iter().all(|event| *event != self.completion_event));
        match &mut self.operation {
            PreparedOperation::Trapdoor { command, rng, .. } => {
                command.submit(std::array::from_fn(|_| {
                    mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes(rng.random())
                }))
            }
            PreparedOperation::Preimage { command, rng, secret, input: None, .. } => {
                command.submit(secret, rng.random())
            }
            PreparedOperation::Preimage { .. } => {
                Err("prepared preimage requires typed trapdoor input".into())
            }
            PreparedOperation::Threshold { command, .. } => command.submit(),
            PreparedOperation::ScalarOp { command, .. } => command.submit(),
            PreparedOperation::ScalarMatrixSelect { command, .. } => command.submit(),
            PreparedOperation::ScalarUpload { .. } => {
                Err("scalar upload requires runtime input".into())
            }
            PreparedOperation::ScalarPack { command, .. } => command.submit(),
            PreparedOperation::InputCopy { command, input, source, .. } => {
                let source = if let Some(source) = source.as_ref() {
                    source.as_ref()
                } else {
                    inputs
                        .get(input.ok_or_else(|| {
                            "prepared input copy source is unavailable".to_owned()
                        })?)
                        .ok_or_else(|| "prepared input copy source is unavailable".to_owned())?
                };
                command.submit_borrowed(source)
            }
            PreparedOperation::Arithmetic { command, .. } => command.submit(),
            PreparedOperation::Accumulate { command, .. } => command.submit(),
            PreparedOperation::Transform { command, target, .. } => command.submit_shared(target),
            PreparedOperation::Modulus { command, target, .. } => command.submit(target),
            PreparedOperation::Transpose { command, .. } => command.submit(),
            PreparedOperation::ConcatRows { commands, sources, .. } => {
                for (command, source) in commands.iter().zip(sources.iter()) {
                    command.submit(Arc::clone(source)).map(|_| ())?;
                }
                Ok(())
            }
            PreparedOperation::CenteredRebase { command, .. } => command.submit(),
            PreparedOperation::GadgetDecompose { command, .. } => command.submit(),
            PreparedOperation::Sampling { command, seed, .. } => command.submit(*seed).map(|_| ()),
            PreparedOperation::CrtRecompose { command, levels, .. } => {
                command.submit(Arc::clone(levels)).map(|_| ())
            }
            PreparedOperation::Alias { .. } | PreparedOperation::LoopBody { .. } => Ok(()),
            PreparedOperation::Selection { candidates, selected, .. } => {
                let index = selected.load(Ordering::Acquire);
                let candidate = candidates
                    .get(index)
                    .ok_or_else(|| "prepared selection candidate is unavailable".to_owned())?;
                candidate.command.submit_borrowed(&candidate.source)
            }
            PreparedOperation::SmallRhs { command, input, source, .. } => {
                if let Some(source) = source {
                    command.submit(Arc::clone(source)).map(|_| ())
                } else {
                    let source = inputs
                        .get(*input)
                        .ok_or("prepared compact multiplication source is unavailable")?;
                    command.submit_borrowed(source)
                }
            }
            PreparedOperation::HashSample { .. } => {
                Err("prepared hash sample requires typed runtime bytes".into())
            }
            PreparedOperation::CompactDecompose { command, .. } => command.submit(),
            PreparedOperation::Reconstruction { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Readback { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Upload { .. } => {
                Err("prepared RNS upload requires typed runtime bytes".into())
            }
        }
    }

    pub(crate) fn submit_borrowed(&mut self, inputs: &[&GpuDCRTPolyMatrix]) -> Result<(), String> {
        self.begin_schedule()?;
        let result = self.submit_borrowed_operation(inputs);
        self.submit_scheduled(result)
    }

    fn submit_runtime_operation(&mut self, inputs: &[PreparedRuntimeValue]) -> Result<(), String> {
        match &mut self.operation {
            PreparedOperation::Preimage { command, rng, secret, input, .. } => {
                let secret = if let Some((input, replica)) = input {
                    let PreparedRuntimeValue::Trapdoor { secret, .. } = &inputs[*input] else {
                        return Err("prepared preimage input has no secret trapdoor".into());
                    };
                    secret.values.get(*replica).ok_or("prepared trapdoor replica is missing")?
                } else {
                    secret
                };
                command.submit(secret, rng.random())
            }
            PreparedOperation::ScalarUpload { command, input, .. } => {
                stage_runtime_scalar(command, &inputs[*input])
            }
            PreparedOperation::InputCopy { command, input, source, .. } => {
                let source = if let Some(source) = source.as_ref() {
                    source.as_ref()
                } else {
                    runtime_matrix_input(
                        inputs,
                        input.ok_or_else(|| {
                            "prepared input copy source is unavailable".to_owned()
                        })?,
                    )?
                };
                command.submit_borrowed(source)
            }
            PreparedOperation::SmallRhs { command, input, source, .. } => {
                if let Some(source) = source {
                    command.submit(Arc::clone(source)).map(|_| ())
                } else {
                    command.submit_borrowed(runtime_matrix_input(inputs, *input)?)
                }
            }
            PreparedOperation::HashSample {
                command,
                input,
                operand_inputs,
                tag_prefix,
                tag_scratch,
                ..
            } => {
                let key =
                    match inputs.get(*input).ok_or("prepared hash key input is unavailable")? {
                        PreparedRuntimeValue::Bytes(bytes) if bytes.len() == 32 => {
                            let mut key = [0u8; 32];
                            key.copy_from_slice(bytes);
                            key
                        }
                        PreparedRuntimeValue::Bytes(_) => {
                            return Err("prepared hash key must contain exactly 32 bytes".into());
                        }
                        _ => return Err("prepared hash key is not a byte input".into()),
                    };
                if operand_inputs.is_empty() {
                    command.submit_key(key).map(|_| ())
                } else {
                    tag_scratch.clear();
                    tag_scratch.extend_from_slice(tag_prefix);
                    for operand_input in operand_inputs {
                        let PreparedRuntimeValue::Int(value) = inputs
                            .get(*operand_input)
                            .ok_or("prepared hash operand is unavailable")?
                        else {
                            return Err("prepared hash operand is not an integer".into());
                        };
                        tag_scratch.push(1);
                        append_hash_tag_integer(tag_scratch, value);
                    }
                    command.submit_key_with_tag(key, tag_scratch).map(|_| ())
                }
            }
            PreparedOperation::CompactDecompose { command, .. } => command.submit(),
            PreparedOperation::Reconstruction { command, in_flight, .. } => {
                *in_flight = Some(command.submit()?);
                Ok(())
            }
            PreparedOperation::Upload { command, input, mode, in_flight, .. } => {
                *in_flight = Some(match mode {
                    PreparedUploadMode::Bytes => {
                        let PreparedRuntimeValue::Bytes(bytes) =
                            inputs.get(*input).ok_or("prepared RNS upload input is unavailable")?
                        else {
                            return Err("prepared RNS upload input is not bytes".into());
                        };
                        command.submit(bytes)?
                    }
                    PreparedUploadMode::Constant => {
                        let PreparedRuntimeValue::Int(value) = inputs
                            .get(*input)
                            .ok_or("prepared constant upload input is unavailable")?
                        else {
                            return Err("prepared constant upload input is not an integer".into());
                        };
                        command.submit_constant(value)?
                    }
                });
                Ok(())
            }
            _ => self.submit_borrowed_operation(&[]),
        }
    }

    fn submit_runtime(&mut self, inputs: &[PreparedRuntimeValue]) -> Result<(), String> {
        self.begin_schedule()?;
        let result = self.submit_runtime_operation(inputs);
        self.submit_scheduled(result)
    }

    pub(crate) fn output(&self) -> (Arc<GpuDCRTPolyMatrix>, i32, usize) {
        match &self.operation {
            PreparedOperation::ScalarPack { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::Threshold { .. } => {
                panic!("device scalar output has no matrix output")
            }
            PreparedOperation::ScalarOp { .. } | PreparedOperation::ScalarUpload { .. } => {
                panic!("scalar command has no matrix output")
            }
            PreparedOperation::InputCopy { output, device, start, .. } |
            PreparedOperation::Trapdoor { output, device, start, .. } |
            PreparedOperation::Arithmetic { output, device, start, .. } |
            PreparedOperation::Accumulate { output, device, start, .. } |
            PreparedOperation::Sampling { output, device, start, .. } |
            PreparedOperation::SmallRhs { output, device, start, .. } |
            PreparedOperation::HashSample { output, device, start, .. } |
            PreparedOperation::CrtRecompose { output, device, start, .. } |
            PreparedOperation::Alias { output, device, start } |
            PreparedOperation::ScalarMatrixSelect { output, device, start, .. } |
            PreparedOperation::Selection { output, device, start, .. } |
            PreparedOperation::LoopBody { output, device, start } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::CompactDecompose { .. } | PreparedOperation::Preimage { .. } => {
                panic!("compact output has no matrix output")
            }
            PreparedOperation::Reconstruction { .. } => {
                panic!("host reconstruction has no matrix output")
            }
            PreparedOperation::Readback { .. } => {
                panic!("host readback has no matrix output")
            }
            PreparedOperation::Upload { target, device, start, .. } => {
                (Arc::clone(target), *device, *start)
            }
            PreparedOperation::Transform { target, device, start, .. } |
            PreparedOperation::Modulus { target, device, start, .. } => {
                (Arc::clone(target), *device, *start)
            }
            PreparedOperation::Transpose { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::ConcatRows { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
            PreparedOperation::CenteredRebase { output, device, start, .. } |
            PreparedOperation::GadgetDecompose { output, device, start, .. } => {
                (Arc::clone(output), *device, *start)
            }
        }
    }

    fn small_output(&self) -> Option<(Arc<GpuSmallMatrix>, i32, usize)> {
        match &self.operation {
            PreparedOperation::CompactDecompose { output, device, start, .. } |
            PreparedOperation::Preimage { output, device, start, .. } => {
                Some((Arc::clone(output), *device, *start))
            }
            _ => None,
        }
    }

    fn wait_until_ready(&mut self) -> Result<(), PreparedCommandError> {
        match &mut self.operation {
            PreparedOperation::Preimage { command, .. } => {
                command.wait().map_err(PreparedCommandError::from_preimage)?;
            }
            PreparedOperation::ScalarUpload { command, .. } => {
                command.wait().map_err(PreparedCommandError::Gpu)?
            }
            PreparedOperation::ScalarOp { command, .. } => {
                command.output().wait().map_err(PreparedCommandError::Gpu)?
            }
            PreparedOperation::Threshold { command, .. } => {
                command.output().wait().map_err(PreparedCommandError::Gpu)?
            }
            PreparedOperation::Reconstruction { in_flight, values, .. } => {
                if let Some(in_flight) = in_flight.take() {
                    in_flight.wait().map_err(PreparedCommandError::Gpu)?;
                    let command_values = in_flight.with_values(|values| values.to_vec());
                    let mut output = values.lock().map_err(|_| {
                        PreparedCommandError::Gpu(
                            "prepared reconstruction result lock poisoned".into(),
                        )
                    })?;
                    output.clone_from_slice(&command_values);
                }
            }
            PreparedOperation::Readback { in_flight, values, .. } => {
                if let Some(in_flight) = in_flight.take() {
                    let command_values =
                        in_flight.wait().map_err(PreparedCommandError::Gpu)?.to_vec();
                    let mut output = values.lock().map_err(|_| {
                        PreparedCommandError::Gpu("prepared readback result lock poisoned".into())
                    })?;
                    output.clone_from_slice(&command_values);
                }
            }
            PreparedOperation::Upload { in_flight, .. } => {
                if let Some(in_flight) = in_flight.take() {
                    in_flight.wait().map_err(PreparedCommandError::Gpu)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn is_ready(&mut self) -> Result<bool, PreparedCommandError> {
        let schedule_ready = self
            .schedule
            .as_ref()
            .map_or(Ok(true), |schedule| schedule.is_ready().map_err(PreparedCommandError::Gpu));
        let operation_ready = match &mut self.operation {
            PreparedOperation::Preimage { command, .. } => {
                command.is_ready().map_err(PreparedCommandError::from_preimage)
            }
            _ => Ok(true),
        }?;
        Ok(schedule_ready? && operation_ready)
    }

    fn reconstruction_output(
        &self,
    ) -> Option<(u32, usize, Arc<Mutex<Box<[num_bigint::BigUint]>>>)> {
        match &self.operation {
            PreparedOperation::Reconstruction { node, start, values, .. } => {
                Some((*node, *start, Arc::clone(values)))
            }
            _ => None,
        }
    }

    fn readback_output(&self) -> Option<(u32, usize, Arc<Mutex<Box<[u64]>>>)> {
        match &self.operation {
            PreparedOperation::Readback { node, start, values, .. } => {
                Some((*node, *start, Arc::clone(values)))
            }
            _ => None,
        }
    }
}

fn runtime_matrix_input(
    inputs: &[PreparedRuntimeValue],
    mut index: usize,
) -> Result<&GpuDCRTPolyMatrix, String> {
    for input in inputs {
        let matrix = match input {
            PreparedRuntimeValue::FleetMatrix(matrix) |
            PreparedRuntimeValue::Trapdoor { public: matrix, .. } => matrix,
            _ => continue,
        };
        if index < matrix.shards().len() {
            return Ok(matrix.shards()[index].value.as_ref());
        }
        index -= matrix.shards().len();
    }
    Err("prepared matrix input source is unavailable".into())
}

fn append_hash_tag_integer(tag: &mut Vec<u8>, value: &num_bigint::BigInt) {
    use num_bigint::Sign;
    let (sign, bytes) = value.to_bytes_be();
    tag.push(match sign {
        Sign::Minus => 1,
        Sign::NoSign | Sign::Plus => 0,
    });
    tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    tag.extend_from_slice(&bytes);
}

struct FleetInstanceState {
    commands: Box<[PreparedCommand]>,
    replay_steps: Arc<[super::gpu_prepared_control::PreparedExecutableCommand]>,
    output_commands: Box<[usize]>,
    small_output_commands: Box<[usize]>,
    input_values: Box<[PreparedRuntimeValue]>,
    scalar_inputs: Box<[super::gpu_prepared_lowering::ScalarValue]>,
    scalar_slots: Arc<[super::gpu_prepared_lowering::ScalarValue]>,
    control_scratch: Box<[super::gpu_prepared_lowering::ScalarValue]>,
    control_results: Box<[super::gpu_prepared_lowering::ScalarValue]>,
    selection_results: Box<[usize]>,
    poisoned: bool,
}

#[derive(Clone, Debug)]
struct PreparedRuntimeInputDescriptor {
    root_index: usize,
    path: Box<[usize]>,
    scalar_slot: Option<usize>,
    max_words: usize,
}

struct FleetInstance {
    state: Mutex<FleetInstanceState>,
}

fn append_replay_step(
    step: &super::gpu_prepared_lowering::PreparedReplayStep,
    program: &super::gpu_prepared_lowering::PreparedProgram,
    commands: &[PreparedCommand],
    native_used: &mut [bool],
    control_by_node: &BTreeMap<u32, usize>,
    output: &mut Vec<super::gpu_prepared_control::PreparedExecutableCommand>,
) -> Result<(), String> {
    match step {
        super::gpu_prepared_lowering::PreparedReplayStep::Node(node_id) => {
            let topology = program
                .topology
                .nodes
                .iter()
                .find(|node| node.id == *node_id)
                .ok_or("prepared replay references an unknown topology node")?;
            if let Some(index) = control_by_node.get(node_id).copied() {
                output.push(super::gpu_prepared_control::PreparedExecutableCommand::Control(index));
            }
            let mut found_native = false;
            for (index, command) in commands.iter().enumerate() {
                if native_used[index] || command.completion_event != topology.completion {
                    continue;
                }
                native_used[index] = true;
                let variant_indices = program
                    .node_sources
                    .get(node_id)
                    .map(|source| source.variant_indices.clone())
                    .unwrap_or_default();
                output.push(super::gpu_prepared_control::PreparedExecutableCommand::Native {
                    index,
                    variant: command.variant,
                    variant_indices,
                });
                found_native = true;
            }
            if matches!(
                topology.command.operation,
                super::gpu_prepared_lowering::PreparedOperation::Gpu(_)
            ) && !found_native
            {
                return Err(format!(
                    "prepared replay node {node_id} has no command for completion event {}",
                    topology.completion
                ));
            }
            if control_by_node.get(node_id).is_none() && !found_native {
                return Err(format!("prepared replay node {node_id} has no fixed command"));
            }
        }
        super::gpu_prepared_lowering::PreparedReplayStep::Sequential {
            count,
            counts,
            offsets,
            banks,
            tail,
        } => {
            let mut converted = [Vec::new(), Vec::new()];
            for bank in 0..2 {
                for nested in banks[bank].iter() {
                    append_replay_step(
                        nested,
                        program,
                        commands,
                        native_used,
                        control_by_node,
                        &mut converted[bank],
                    )?;
                }
            }
            let mut converted_tail = Vec::new();
            for nested in tail.iter() {
                append_replay_step(
                    nested,
                    program,
                    commands,
                    native_used,
                    control_by_node,
                    &mut converted_tail,
                )?;
            }
            output.push(super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                count: *count,
                counts: counts.clone(),
                offsets: offsets.clone(),
                banks: [
                    converted[0].clone().into_boxed_slice(),
                    converted[1].clone().into_boxed_slice(),
                ],
                tail: converted_tail.into_boxed_slice(),
            });
        }
        super::gpu_prepared_lowering::PreparedReplayStep::Subgraph { body } => {
            let mut converted = Vec::new();
            for nested in body.iter() {
                append_replay_step(
                    nested,
                    program,
                    commands,
                    native_used,
                    control_by_node,
                    &mut converted,
                )?;
            }
            output.push(super::gpu_prepared_control::PreparedExecutableCommand::Subgraph {
                body: converted.into_boxed_slice(),
            });
        }
        super::gpu_prepared_lowering::PreparedReplayStep::Parallel { counts, waves } => {
            let mut converted_waves = Vec::with_capacity(waves.len());
            for wave in waves.iter() {
                let mut converted = Vec::new();
                for nested in wave.iter() {
                    append_replay_step(
                        nested,
                        program,
                        commands,
                        native_used,
                        control_by_node,
                        &mut converted,
                    )?;
                }
                converted_waves.push(converted.into_boxed_slice());
            }
            output.push(super::gpu_prepared_control::PreparedExecutableCommand::Parallel {
                counts: counts.clone(),
                waves: converted_waves.into_boxed_slice(),
            });
        }
    }
    Ok(())
}

struct PreparedGpuSlotPool {
    instances: Box<[Arc<FleetInstance>]>,
    region: Arc<crate::gpu_memory::GpuMemoryRegion>,
    free_mask: AtomicUsize,
    poisoned: AtomicUsize,
    retired: Mutex<Vec<usize>>,
}

impl PreparedGpuSlotPool {
    fn acquire(&self) -> Result<usize, PreparedGpuRunError> {
        debug_assert!(Arc::strong_count(&self.region) > 0);
        self.reclaim_retired()?;
        loop {
            let mask = self.free_mask.load(Ordering::Acquire);
            if mask == 0 {
                return if self.poisoned.load(Ordering::Acquire) == self.instances.len() {
                    Err(PreparedGpuRunError::Failed(
                        "all prepared GPU instances have failed".into(),
                    ))
                } else {
                    Err(PreparedGpuRunError::Busy(PreparedGpuBusy))
                };
            }
            let slot = mask.trailing_zeros() as usize;
            let bit = 1usize << slot;
            if self
                .free_mask
                .compare_exchange(mask, mask & !bit, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(slot);
            }
        }
    }

    fn retire(&self, slot: usize) {
        if let Ok(mut retired) = self.retired.lock() {
            retired.push(slot);
        } else {
            self.poisoned.fetch_add(1, Ordering::Release);
        }
    }

    fn reclaim_retired(&self) -> Result<(), PreparedGpuRunError> {
        // Poll every retired slot without waiting. Keep pending slots retired;
        // a later acquire will poll them again. The queue lock is held only
        // while transferring ownership of the candidate list, never across
        // CUDA synchronization or command-state polling.
        let retired_slots = {
            let mut retired = self.retired.lock().map_err(|_| {
                PreparedGpuRunError::Failed("prepared retirement state poisoned".into())
            })?;
            std::mem::take(&mut *retired)
        };
        let mut pending = Vec::new();
        let mut first_error = None;
        for slot in retired_slots {
            let instance = &self.instances[slot];
            let mut state = match instance.state.lock() {
                Ok(state) => state,
                Err(_) => {
                    self.poisoned.fetch_add(1, Ordering::Release);
                    first_error.get_or_insert_with(|| {
                        PreparedGpuRunError::Failed("prepared GPU instance poisoned".into())
                    });
                    continue;
                }
            };
            if state.poisoned {
                self.poisoned.fetch_add(1, Ordering::Release);
                first_error.get_or_insert_with(|| {
                    PreparedGpuRunError::Failed("prepared GPU instance poisoned".into())
                });
                continue;
            }
            let ready = state.commands.iter_mut().try_fold(true, |ready, command| {
                Ok::<_, PreparedCommandError>(ready && command.is_ready()?)
            });
            match ready {
                Ok(true) => {
                    self.free_mask.fetch_or(1usize << slot, Ordering::Release);
                }
                Ok(false) => pending.push(slot),
                Err(error) => {
                    state.poisoned = true;
                    self.poisoned.fetch_add(1, Ordering::Release);
                    first_error.get_or_insert(PreparedGpuRunError::from_command_error(error));
                }
            }
        }
        if !pending.is_empty() {
            let mut retired = self.retired.lock().map_err(|_| {
                PreparedGpuRunError::Failed("prepared retirement state poisoned".into())
            })?;
            retired.extend(pending);
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedGpuBusy;

impl std::fmt::Display for PreparedGpuBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("all prepared GPU execution instances are busy")
    }
}
impl std::error::Error for PreparedGpuBusy {}

#[derive(Debug)]
pub enum PreparedGpuRunError {
    Busy(PreparedGpuBusy),
    Failed(String),
    SamplingExhausted { column_start: usize, column_count: usize, attempts: usize },
}

impl PreparedGpuRunError {
    fn from_command_error(error: PreparedCommandError) -> Self {
        match error {
            PreparedCommandError::SamplingExhausted { column_start, column_count, attempts } => {
                Self::SamplingExhausted { column_start, column_count, attempts }
            }
            PreparedCommandError::Gpu(error) => Self::Failed(error),
        }
    }
}

impl std::fmt::Display for PreparedGpuRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy(error) => error.fmt(f),
            Self::Failed(error) => f.write_str(error),
            Self::SamplingExhausted { column_start, column_count, attempts } => write!(
                f,
                "prepared preimage sampling exhausted at columns {column_start}..{} after {attempts} attempts",
                column_start + column_count
            ),
        }
    }
}
impl std::error::Error for PreparedGpuRunError {}

pub struct PreparedGpuFleetOutput {
    pool: Arc<PreparedGpuSlotPool>,
    slot: usize,
    rows: usize,
    columns: usize,
    pub(crate) output_descriptors: Arc<[PreparedGpuOutputDescriptor]>,
    pub(crate) output_names: Arc<[String]>,
    output_indices: Arc<BTreeMap<String, usize>>,
    scalar_output_commands: Arc<BTreeMap<WireRef, usize>>,
    host_output_descriptors: Arc<[PreparedHostOutputDescriptor]>,
    scalar_slots: Option<Arc<[super::gpu_prepared_lowering::ScalarValue]>>,
}

#[derive(Clone, Debug)]
pub(crate) enum PreparedGpuOutputKind {
    Trapdoor {
        indices: Box<[usize]>,
        matrix_type: mxx_ir_core::types::ConcreteMatrixType,
        sigma: f64,
        gadget_base: num_bigint::BigInt,
        digit_count: usize,
    },
    Matrix(Box<[usize]>),
    MatrixOrdinal(usize),
    SmallMatrix,
    Family(Box<[WireRef]>),
    Host(usize),
    Scalar {
        wire: WireRef,
        control: Option<usize>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedGpuOutputDescriptor {
    pub(crate) kind: PreparedGpuOutputKind,
}

#[derive(Clone, Copy, Debug)]
enum PreparedHostOutputKind {
    Reconstruction,
    Readback,
}

#[derive(Clone, Debug)]
struct PreparedHostOutputDescriptor {
    kind: PreparedHostOutputKind,
    commands: Box<[usize]>,
}

impl PreparedGpuFleetOutput {
    pub(crate) fn materialize_scalar_output(
        &self,
        wire: WireRef,
        host_slot: Option<usize>,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        use super::gpu_prepared_lowering::ScalarValue;
        use crate::backend::RuntimeValue;
        self.wait_until_ready()?;
        let value = self
            .device_scalar_output(wire)
            .map_err(|error| error.to_string())?
            .or_else(|| host_slot.and_then(|slot| self.scalar_slot_values().get(slot)).cloned())
            .ok_or_else(|| "prepared scalar output is not bound".to_owned())?;
        match value {
            ScalarValue::Int(value) => Ok(RuntimeValue::Int(value)),
            ScalarValue::Real(value) => Ok(RuntimeValue::Real(value)),
            ScalarValue::Bool(value) => Ok(RuntimeValue::Bool(value)),
            ScalarValue::Slot(_) | ScalarValue::Runtime(_) => {
                Err("prepared scalar output is not materialized".into())
            }
        }
    }
}

impl std::fmt::Debug for PreparedGpuFleetOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedGpuFleetOutput")
            .field("slot", &self.slot)
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .finish()
    }
}

impl PreparedGpuFleetOutput {
    pub(crate) fn check_device_scalar_status(&self) -> Result<(), String> {
        let state = self.pool.instances[self.slot]
            .state
            .lock()
            .map_err(|_| "prepared output state poisoned")?;
        for command in state.commands.iter() {
            match &command.operation {
                PreparedOperation::ScalarPack { command, .. } => command.check_sources()?,
                PreparedOperation::ScalarMatrixSelect { command, .. } => {
                    command.check_selector()?
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub fn control_results(&self) -> Box<[super::gpu_prepared_lowering::ScalarValue]> {
        let state =
            self.pool.instances[self.slot].state.lock().expect("prepared GPU instance poisoned");
        state.control_results.clone()
    }

    pub(crate) fn scalar_slot_values(&self) -> &[super::gpu_prepared_lowering::ScalarValue] {
        self.scalar_slots.as_deref().expect("live prepared output")
    }

    pub(crate) fn device_scalar_output(
        &self,
        wire: WireRef,
    ) -> Result<Option<super::gpu_prepared_lowering::ScalarValue>, String> {
        let state =
            self.pool.instances[self.slot].state.lock().expect("prepared GPU instance poisoned");
        let Some(command_index) = self.scalar_output_commands.get(&wire).copied() else {
            return Ok(None);
        };
        let entry = state
            .commands
            .get(command_index)
            .ok_or_else(|| "prepared scalar output command index is invalid".to_owned())?;
        match &entry.operation {
            PreparedOperation::ScalarOp { command, kind, .. } => command
                .output()
                .with_words(|words, _| {
                    use super::gpu_prepared_lowering::ScalarValue;
                    use mxx_ir_core::types::ConcreteWireType;
                    match kind {
                        ConcreteWireType::Bool | ConcreteWireType::ConstantBool => {
                            ScalarValue::Bool(words[0] != 0)
                        }
                        ConcreteWireType::Real | ConcreteWireType::ConstantReal => {
                            ScalarValue::Real(f64::from_bits(words[0]))
                        }
                        _ => ScalarValue::Int(num_bigint::BigInt::from_signed_bytes_le(
                            &words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>(),
                        )),
                    }
                })
                .map(Some),
            PreparedOperation::ScalarUpload { command, kind, .. } => command
                .with_words(|words, _| {
                    use super::gpu_prepared_lowering::ScalarValue;
                    use mxx_ir_core::types::ConcreteWireType;
                    match kind {
                        ConcreteWireType::Bool | ConcreteWireType::ConstantBool => {
                            ScalarValue::Bool(words[0] != 0)
                        }
                        ConcreteWireType::Real | ConcreteWireType::ConstantReal => {
                            ScalarValue::Real(f64::from_bits(words[0]))
                        }
                        _ => ScalarValue::Int(num_bigint::BigInt::from_signed_bytes_le(
                            &words.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>(),
                        )),
                    }
                })
                .map(Some),
            PreparedOperation::Threshold { command, output_bool, .. } => command
                .with_words(|words, width| {
                    let words =
                        &words[wire.port.0 as usize * width..(wire.port.0 as usize + 1) * width];
                    if *output_bool {
                        super::gpu_prepared_lowering::ScalarValue::Bool(words[0] != 0)
                    } else {
                        let digits = words
                            .iter()
                            .flat_map(|word| [*word as u32, (*word >> 32) as u32])
                            .collect::<Vec<_>>();
                        super::gpu_prepared_lowering::ScalarValue::Int(
                            num_bigint::BigUint::from_slice(&digits).into(),
                        )
                    }
                })
                .map(Some),
            _ => Err("prepared scalar output command has an incompatible operation".to_owned()),
        }
    }

    pub(crate) fn wait_until_ready_typed(&self) -> Result<(), PreparedGpuRunError> {
        let instance = &self.pool.instances[self.slot];
        let mut state = instance
            .state
            .lock()
            .map_err(|_| PreparedGpuRunError::Failed("prepared GPU instance poisoned".into()))?;
        for command in state.commands.iter_mut() {
            command.wait_until_ready().map_err(PreparedGpuRunError::from_command_error)?
        }
        for index in state.output_commands.iter().copied() {
            state.commands[index].output().0.wait_until_ready();
        }
        Ok(())
    }

    pub fn wait_until_ready(&self) -> Result<(), String> {
        self.wait_until_ready_typed().map_err(|error| error.to_string())
    }

    pub fn materialize(&self) -> Result<GpuFleetMatrix, String> {
        self.wait_until_ready()?;
        Ok(self.materialize_async())
    }

    /// Build the public fleet owner without synchronizing the submitted
    /// commands. The retained prepared lease keeps every instance owner and
    /// its completion events alive; normal matrix consumers wait through the
    /// owner event chain when they actually read it.
    pub(crate) fn materialize_async(&self) -> GpuFleetMatrix {
        let instance = &self.pool.instances[self.slot];
        let state = instance.state.lock().expect("prepared GPU instance poisoned");
        GpuFleetMatrix::from_shared_shards(
            self.rows,
            self.columns,
            state
                .output_commands
                .iter()
                .map(|index| {
                    let command = &state.commands[*index];
                    let (value, device_id, global_column_start) = command.output();
                    GpuColumnShard { device_id, global_column_start, value: Arc::clone(&value) }
                })
                .collect(),
        )
    }

    pub(crate) fn materialize_output(&self, indices: &[usize]) -> Result<GpuFleetMatrix, String> {
        let instance = &self.pool.instances[self.slot];
        let state =
            instance.state.lock().map_err(|_| "prepared output state poisoned".to_owned())?;
        let (rows, columns) = indices
            .first()
            .map(|index| state.commands[*index].output().0.size())
            .ok_or_else(|| "prepared output binding has no shards".to_owned())?;
        Ok(GpuFleetMatrix::from_shared_shards(
            rows,
            columns,
            indices
                .iter()
                .map(|index| {
                    let command = &state.commands[*index];
                    let (value, device_id, global_column_start) = command.output();
                    GpuColumnShard { device_id, global_column_start, value: Arc::clone(&value) }
                })
                .collect(),
        ))
    }

    fn materialize_host_output(
        &self,
        descriptor: usize,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, String> {
        record_prepared_output_reconstruction();
        let descriptor = self
            .host_output_descriptors
            .get(descriptor)
            .ok_or_else(|| "prepared host output descriptor is out of bounds".to_owned())?;
        let state = self.pool.instances[self.slot]
            .state
            .lock()
            .map_err(|_| "prepared host output state poisoned".to_owned())?;
        record_prepared_host_allocation();
        match descriptor.kind {
            PreparedHostOutputKind::Reconstruction => {
                let mut values = Vec::new();
                for index in descriptor.commands.iter().copied() {
                    let (_, _, output) =
                        state.commands[index].reconstruction_output().ok_or_else(|| {
                            "prepared reconstruction descriptor is invalid".to_owned()
                        })?;
                    let output = output
                        .lock()
                        .map_err(|_| "prepared host reconstruction is poisoned".to_owned())?;
                    values.extend(output.iter().cloned().map(|value| {
                        crate::backend::RuntimeValue::Int(num_bigint::BigInt::from(value))
                    }));
                }
                Ok(crate::backend::RuntimeValue::IndexedFamily(values))
            }
            PreparedHostOutputKind::Readback => {
                let mut values = Vec::new();
                for index in descriptor.commands.iter().copied() {
                    let (_, _, output) = state.commands[index]
                        .readback_output()
                        .ok_or_else(|| "prepared readback descriptor is invalid".to_owned())?;
                    let output = output
                        .lock()
                        .map_err(|_| "prepared host readback is poisoned".to_owned())?;
                    values.extend(output.iter().copied().map(|value| {
                        crate::backend::RuntimeValue::Int(num_bigint::BigInt::from(value))
                    }));
                }
                Ok(crate::backend::RuntimeValue::IndexedFamily(values))
            }
        }
    }

    pub fn materialize_small(&self) -> Result<GpuFleetSmallMatrix, String> {
        self.wait_until_ready()?;
        Ok(self.materialize_small_async())
    }

    pub(crate) fn materialize_small_async(&self) -> GpuFleetSmallMatrix {
        let instance = &self.pool.instances[self.slot];
        let state = instance.state.lock().expect("prepared GPU instance poisoned");
        let (rows, columns) = state
            .small_output_commands
            .first()
            .and_then(|index| state.commands[*index].small_output())
            .map(|(value, _, _)| value.size())
            .unwrap_or((self.rows, self.columns));
        GpuFleetSmallMatrix::from_shared_shards(
            rows,
            columns,
            state
                .small_output_commands
                .iter()
                .map(|index| {
                    let command = &state.commands[*index];
                    let (value, device_id, global_column_start) = command
                        .small_output()
                        .expect("prepared small output command missing compact owner");
                    GpuColumnShard { device_id, global_column_start, value }
                })
                .collect(),
        )
    }
}

impl Drop for PreparedGpuFleetOutput {
    fn drop(&mut self) {
        // The pool bit is a reuse permission, not merely an ownership count.
        // Retiring it before the final writer event completes lets the next
        // instance overwrite a matrix still observed by a GPU reader.
        // Retire the scalar snapshot before publishing the instance reuse bit.
        // The next submit then has exclusive access to its preallocated array.
        drop(self.scalar_slots.take());
        self.pool.retire(self.slot);
    }
}

impl crate::executor::PreparedOutputLease<GpuDcrtBackend> for Arc<PreparedGpuFleetOutput> {
    fn output_names(&self) -> &[String] {
        self.output_names.as_ref()
    }

    fn materialize(
        &self,
        name: &str,
        _backend: &mut GpuDcrtBackend,
    ) -> Result<crate::backend::RuntimeValue<GpuDcrtBackend>, crate::executor::ExecutionError> {
        let descriptor_index = self
            .output_indices
            .get(name)
            .copied()
            .ok_or_else(|| crate::executor::ExecutionError::MissingOutput(name.to_owned()))?;
        let descriptor = self.output_descriptors.get(descriptor_index).ok_or_else(|| {
            crate::executor::ExecutionError::Backend("prepared output index is invalid".into())
        })?;
        let backend_error = |error: String| crate::executor::ExecutionError::Backend(error);
        let prepared_error = |error: PreparedGpuRunError| match error {
            PreparedGpuRunError::SamplingExhausted { column_start, column_count, attempts } => {
                crate::executor::ExecutionError::SamplingExhausted {
                    column_start,
                    column_end: column_start + column_count,
                    attempts,
                }
            }
            error => backend_error(error.to_string()),
        };
        // Sampling rejection is only known on the device. Resolve it at this
        // explicit boundary, including when the exported value is downstream.
        self.wait_until_ready_typed().map_err(prepared_error)?;
        match &descriptor.kind {
            PreparedGpuOutputKind::Trapdoor {
                indices,
                matrix_type,
                sigma,
                gadget_base,
                digit_count,
            } => {
                let public = self.materialize_output(indices).map_err(backend_error)?;
                let state = self.pool.instances[self.slot]
                    .state
                    .lock()
                    .map_err(|_| backend_error("prepared trapdoor state poisoned".into()))?;
                let values = indices
                    .iter()
                    .map(|index| match &state.commands[*index].operation {
                        PreparedOperation::Trapdoor { command, .. } => {
                            Ok(Arc::clone(command.trapdoor()))
                        }
                        _ => Err(backend_error(
                            "prepared trapdoor output has no secret owner".into(),
                        )),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(crate::backend::RuntimeValue::Trapdoor {
                    secret: Some(Arc::new(super::GpuFleetTrapdoor {
                        values: Arc::new(values),
                        prepared_lease: Some(Arc::clone(self)),
                    })),
                    public: Arc::new(GpuFleetMatrix::with_prepared_lease(public, Arc::clone(self))),
                    matrix_type: matrix_type.clone(),
                    sigma: *sigma,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                    gadget_small: None,
                })
            }
            PreparedGpuOutputKind::Matrix(indices) => {
                let output = self.materialize_output(indices).map_err(backend_error)?;
                Ok(crate::backend::RuntimeValue::Matrix(Arc::new(
                    GpuFleetMatrix::with_prepared_lease(output, Arc::clone(self)),
                )))
            }
            PreparedGpuOutputKind::MatrixOrdinal(_) => {
                Err(backend_error("prepared matrix output descriptor was not finalized".into()))
            }
            PreparedGpuOutputKind::SmallMatrix => Ok(crate::backend::RuntimeValue::SmallMatrix(
                Arc::new(GpuFleetSmallMatrix::with_prepared_lease(
                    self.materialize_small_async(),
                    Arc::clone(self),
                )),
            )),
            PreparedGpuOutputKind::Family(members) => {
                let values = members
                    .iter()
                    .map(|wire| self.materialize_scalar_output(*wire, None).map_err(backend_error))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(crate::backend::RuntimeValue::IndexedFamily(values))
            }
            PreparedGpuOutputKind::Host(descriptor) => {
                self.materialize_host_output(*descriptor).map_err(backend_error)
            }
            PreparedGpuOutputKind::Scalar { wire, control } => {
                if let Ok(value) = self.materialize_scalar_output(*wire, None) {
                    return Ok(value);
                }
                self.wait_until_ready_typed().map_err(prepared_error)?;
                let value = self
                    .control_results()
                    .get(control.ok_or_else(|| {
                        backend_error("prepared scalar output has no control result".into())
                    })?)
                    .cloned()
                    .ok_or_else(|| {
                        backend_error("prepared scalar output result is missing".into())
                    })?;
                match value {
                    super::gpu_prepared_lowering::ScalarValue::Int(value) => {
                        Ok(crate::backend::RuntimeValue::Int(value))
                    }
                    super::gpu_prepared_lowering::ScalarValue::Real(value) => {
                        Ok(crate::backend::RuntimeValue::Real(value))
                    }
                    super::gpu_prepared_lowering::ScalarValue::Bool(value) => {
                        Ok(crate::backend::RuntimeValue::Bool(value))
                    }
                    _ => Err(backend_error("prepared scalar output was not resolved".into())),
                }
            }
        }
    }
}

pub struct PreparedGpuFleetExecution {
    pool: Arc<PreparedGpuSlotPool>,
    rows: usize,
    columns: usize,
    program: Option<Arc<super::gpu_prepared_lowering::PreparedProgram>>,
    control_commands: Arc<[super::gpu_prepared_control::PreparedControlCommand]>,
    pub(crate) control_output_indices: Arc<[(String, usize)]>,
    pub(crate) output_control_indices: Arc<[Option<usize>]>,
    scalar_input_max_words: Arc<[usize]>,
    runtime_input_descriptors: Arc<[PreparedRuntimeInputDescriptor]>,
    /// Output ordinals and family members are fixed during warmup.  Execute
    /// must never rediscover these relationships by scanning the graph.
    pub(crate) output_matrix_ordinals: Arc<[Option<usize>]>,
    pub(crate) output_family_members: Arc<[Box<[WireRef]>]>,
    pub(crate) output_descriptors: Arc<[PreparedGpuOutputDescriptor]>,
    pub(crate) output_names: Arc<[String]>,
    output_indices: Arc<BTreeMap<String, usize>>,
    scalar_output_commands: Arc<BTreeMap<WireRef, usize>>,
    host_output_descriptors: Arc<[PreparedHostOutputDescriptor]>,
}

impl PreparedGpuFleetExecution {
    /// Assemble one fixed executable from preparation-time command builders.
    /// Commands own every native plan and destination they reference; the
    /// region is retained by the pool until every output lease retires.
    pub(crate) fn from_command_instances(
        instances: Vec<(Box<[PreparedCommand]>, Box<[usize]>)>,
        region: Arc<crate::gpu_memory::GpuMemoryRegion>,
        rows: usize,
        columns: usize,
    ) -> Self {
        let instances = instances
            .into_iter()
            .map(|(commands, output_commands)| {
                Arc::new(FleetInstance {
                    state: Mutex::new(FleetInstanceState {
                        commands,
                        replay_steps: Arc::from([]),
                        output_commands,
                        small_output_commands: Vec::new().into_boxed_slice(),
                        input_values: Box::new([]),
                        scalar_inputs: Box::new([]),
                        scalar_slots: Arc::from([]),
                        control_scratch: Box::new([]),
                        control_results: Box::new([]),
                        selection_results: Box::new([]),
                        poisoned: false,
                    }),
                })
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let instance_count = instances.len();
        let pool = Arc::new(PreparedGpuSlotPool {
            instances,
            region,
            free_mask: AtomicUsize::new(usize::MAX >> (usize::BITS as usize - instance_count)),
            poisoned: AtomicUsize::new(0),
            retired: Mutex::new(Vec::with_capacity(instance_count)),
        });
        Self {
            pool,
            rows,
            columns,
            program: None,
            control_commands: Arc::from([]),
            control_output_indices: Arc::from([]),
            output_control_indices: Arc::from([]),
            scalar_input_max_words: Arc::from([]),
            runtime_input_descriptors: Arc::from([]),
            output_matrix_ordinals: Arc::from([]),
            output_family_members: Arc::from([]),
            output_descriptors: Arc::from([]),
            output_names: Arc::from([]),
            output_indices: Arc::new(BTreeMap::new()),
            scalar_output_commands: Arc::new(BTreeMap::new()),
            host_output_descriptors: Arc::from([]),
        }
    }

    pub(crate) fn with_small_output_indices(self, indices: Vec<Box<[usize]>>) -> Self {
        for (instance, indices) in self.pool.instances.iter().zip(indices) {
            instance.state.lock().expect("prepared GPU instance poisoned").small_output_commands =
                indices;
        }
        self
    }

    pub(crate) fn with_program(
        mut self,
        program: Arc<super::gpu_prepared_lowering::PreparedProgram>,
    ) -> Result<Self, String> {
        record_prepared_topology_scan(program.topology.nodes.len());
        self.control_commands =
            Arc::from(super::gpu_prepared_control::build_control_commands(&program)?);
        self.scalar_input_max_words = Arc::from(program.scalar_input_max_words.clone());
        self.runtime_input_descriptors = Arc::from(
            program
                .runtime_input_wires
                .iter()
                .enumerate()
                .map(|(index, wire)| PreparedRuntimeInputDescriptor {
                    root_index: program.runtime_input_roots[index],
                    path: program
                        .input_leaf_bindings
                        .get(wire)
                        .map(|binding| binding.path.clone())
                        .unwrap_or_default(),
                    scalar_slot: program.scalar_slots.get(wire).copied(),
                    max_words: program.scalar_input_max_words.get(index).copied().unwrap_or(0),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let mut control_output_indices = Vec::new();
        for (control_index, node) in program
            .topology
            .nodes
            .iter()
            .filter(|node| {
                !matches!(
                    node.command.operation,
                    super::gpu_prepared_lowering::PreparedOperation::Gpu(_)
                )
            })
            .enumerate()
        {
            if !matches!(
                node.command.operation,
                super::gpu_prepared_lowering::PreparedOperation::Scalar
            ) {
                continue;
            }
            for (name, wire) in program.output_names.iter() {
                if wire.node.0 as u32 == node.id {
                    control_output_indices.push((name.clone(), control_index));
                }
            }
        }
        self.control_output_indices = Arc::from(control_output_indices.into_boxed_slice());
        self.output_control_indices = Arc::from(
            program
                .output_bindings
                .iter()
                .map(|binding| {
                    self.control_output_indices
                        .iter()
                        .find(|(name, _)| name == &binding.name)
                        .map(|(_, index)| *index)
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let mut matrix_ordinal = 0usize;
        let mut matrix_ordinals = Vec::with_capacity(program.output_bindings.len());
        let mut family_members = Vec::with_capacity(program.output_bindings.len());
        for binding in program.output_bindings.iter() {
            let ordinal =
                if matches!(binding.kind, super::gpu_prepared_lowering::PreparedOutputKind::Matrix)
                {
                    let ordinal = matrix_ordinal;
                    matrix_ordinal += 1;
                    Some(ordinal)
                } else {
                    None
                };
            matrix_ordinals.push(ordinal);
            let members = program.family_wires.get(&binding.wire).cloned().unwrap_or_else(|| {
                program
                    .node_bindings
                    .get(&(binding.wire.node.0 as u32))
                    .map(|(arguments, _)| {
                        arguments
                            .iter()
                            .filter(|member| program.scalar_slots.contains_key(member))
                            .copied()
                            .collect()
                    })
                    .unwrap_or_default()
            });
            family_members.push(members);
        }
        self.output_matrix_ordinals = Arc::from(matrix_ordinals.into_boxed_slice());
        self.output_family_members = Arc::from(family_members.into_boxed_slice());
        let mut host_output_descriptors = Vec::new();
        let mut host_descriptor_indices = vec![None; program.output_bindings.len()];
        let first_state = self
            .pool
            .instances
            .first()
            .ok_or("prepared output pool is empty")?
            .state
            .lock()
            .map_err(|_| "prepared GPU instance poisoned")?;
        for (index, binding) in program.output_bindings.iter().enumerate() {
            if !matches!(
                binding.kind,
                super::gpu_prepared_lowering::PreparedOutputKind::HostReconstruction
            ) {
                continue;
            }
            let node = binding.wire.node.0 as u32;
            let mut reconstruction = first_state
                .commands
                .iter()
                .enumerate()
                .filter_map(|(command, value)| {
                    value
                        .reconstruction_output()
                        .and_then(|(owner, start, _)| (owner == node).then_some((start, command)))
                })
                .collect::<Vec<_>>();
            let kind = if !reconstruction.is_empty() {
                reconstruction.sort_by_key(|(start, _)| *start);
                PreparedHostOutputKind::Reconstruction
            } else {
                let mut readback = first_state
                    .commands
                    .iter()
                    .enumerate()
                    .filter_map(|(command, value)| {
                        value.readback_output().and_then(|(owner, start, _)| {
                            (owner == node).then_some((start, command))
                        })
                    })
                    .collect::<Vec<_>>();
                if readback.is_empty() {
                    return Err("prepared host output has no fixed command descriptors".into());
                }
                readback.sort_by_key(|(start, _)| *start);
                reconstruction = readback;
                PreparedHostOutputKind::Readback
            };
            let descriptor = host_output_descriptors.len();
            host_descriptor_indices[index] = Some(descriptor);
            host_output_descriptors.push(PreparedHostOutputDescriptor {
                kind,
                commands: reconstruction.into_iter().map(|(_, command)| command).collect(),
            });
        }
        drop(first_state);
        self.host_output_descriptors = Arc::from(host_output_descriptors.into_boxed_slice());
        let mut descriptors = Vec::with_capacity(program.output_bindings.len());
        for (index, binding) in program.output_bindings.iter().enumerate() {
            let kind = match binding.kind {
                super::gpu_prepared_lowering::PreparedOutputKind::Matrix => {
                    PreparedGpuOutputKind::MatrixOrdinal(
                        self.output_matrix_ordinals[index]
                            .ok_or("prepared matrix output ordinal is missing")?,
                    )
                }
                super::gpu_prepared_lowering::PreparedOutputKind::SmallMatrix => {
                    PreparedGpuOutputKind::SmallMatrix
                }
                super::gpu_prepared_lowering::PreparedOutputKind::Family => {
                    PreparedGpuOutputKind::Family(self.output_family_members[index].clone())
                }
                super::gpu_prepared_lowering::PreparedOutputKind::HostReconstruction => {
                    PreparedGpuOutputKind::Host(
                        host_descriptor_indices[index]
                            .ok_or("prepared host output descriptor is missing")?,
                    )
                }
                super::gpu_prepared_lowering::PreparedOutputKind::Scalar => {
                    PreparedGpuOutputKind::Scalar {
                        wire: binding.wire,
                        control: self.output_control_indices[index],
                    }
                }
            };
            descriptors.push(PreparedGpuOutputDescriptor { kind });
        }
        self.output_names = Arc::from(
            program
                .output_bindings
                .iter()
                .map(|binding| binding.name.clone())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );
        let output_count = program
            .outputs
            .iter()
            .filter(|wire| {
                matches!(
                    program.wire_types.get(wire),
                    Some(
                        mxx_ir_core::types::ConcreteWireType::Matrix(_) |
                            mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }
                    )
                )
            })
            .count();
        let matrix_commands = if output_count == 0 {
            Vec::new().into_boxed_slice()
        } else {
            let first = self.pool.instances.first().ok_or("prepared output pool is empty")?;
            let state = first.state.lock().expect("prepared GPU instance poisoned");
            let shard_count = state.output_commands.len() / output_count;
            if shard_count * output_count != state.output_commands.len() {
                return Err("prepared output command grouping is inconsistent".into());
            }
            let expected = shard_count * output_count;
            for instance in self.pool.instances.iter().skip(1) {
                let state = instance.state.lock().expect("prepared GPU instance poisoned");
                if state.output_commands.len() != expected {
                    return Err("prepared output command grouping is inconsistent".into());
                }
            }
            (0..output_count)
                .map(|output| {
                    (0..shard_count)
                        .map(|shard| state.output_commands[shard * output_count + output])
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };
        for descriptor in &mut descriptors {
            if let PreparedGpuOutputKind::MatrixOrdinal(ordinal) = &descriptor.kind {
                descriptor.kind = PreparedGpuOutputKind::Matrix(
                    matrix_commands
                        .get(*ordinal)
                        .cloned()
                        .ok_or("prepared matrix output descriptor is missing")?,
                );
            }
        }
        for (descriptor, binding) in descriptors.iter_mut().zip(program.output_bindings.iter()) {
            if let mxx_ir_core::types::ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                ..
            } = &program.wire_types[&binding.wire]
            {
                let PreparedGpuOutputKind::Matrix(indices) = &descriptor.kind else {
                    return Err("prepared trapdoor output indices missing".into());
                };
                let source = program
                    .node_sources
                    .get(&(binding.wire.node.0 as u32))
                    .ok_or("prepared trapdoor output environment missing")?;
                descriptor.kind = PreparedGpuOutputKind::Trapdoor {
                    indices: indices.clone(),
                    matrix_type: matrix.clone(),
                    sigma: sigma
                        .evaluate_f64(&source.environment)
                        .map_err(|error| error.to_string())?,
                    gadget_base: gadget_base.clone(),
                    digit_count: *digit_count,
                };
            }
        }
        self.output_descriptors = Arc::from(descriptors.into_boxed_slice());
        self.output_indices = Arc::new(
            program
                .output_bindings
                .iter()
                .enumerate()
                .map(|(index, binding)| (binding.name.clone(), index))
                .collect(),
        );
        let scalar_output_commands = {
            let first = self.pool.instances.first().ok_or("prepared output pool is empty")?;
            let state = first.state.lock().expect("prepared GPU instance poisoned");
            program
                .output_bindings
                .iter()
                .filter(|binding| {
                    matches!(binding.kind, super::gpu_prepared_lowering::PreparedOutputKind::Scalar)
                })
                .filter_map(|binding| {
                    let command = state.commands.iter().enumerate().find_map(|(index, entry)| {
                        match &entry.operation {
                            PreparedOperation::ScalarOp { command, wire, .. }
                                if *wire == binding.wire =>
                            {
                                Some(index)
                            }
                            PreparedOperation::ScalarUpload { wire, .. }
                                if *wire == binding.wire =>
                            {
                                Some(index)
                            }
                            PreparedOperation::Threshold { node, .. }
                                if *node == binding.wire.node.0 as u32 =>
                            {
                                Some(index)
                            }
                            _ => None,
                        }
                    });
                    command.map(|command| (binding.wire, command))
                })
                .collect::<BTreeMap<_, _>>()
        };
        self.scalar_output_commands = Arc::new(scalar_output_commands);
        let scratch_len = self
            .control_commands
            .iter()
            .filter_map(|command| match command {
                super::gpu_prepared_control::PreparedControlCommand::Scalar {
                    command,
                    variants,
                    ..
                } => Some(
                    variants
                        .iter()
                        .map(|variant| variant.value_count())
                        .max()
                        .unwrap_or(0)
                        .max(command.value_count()),
                ),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let result_len = self.control_commands.len();
        let selection_len = self.control_commands.len();
        let input_values =
            vec![PreparedRuntimeValue::Bool(false); program.runtime_input_wires.len()]
                .into_boxed_slice();
        for instance in &self.pool.instances {
            let mut state = instance.state.lock().expect("prepared GPU instance poisoned");
            state.input_values = input_values.clone();
            state.scalar_inputs = vec![
                super::gpu_prepared_lowering::ScalarValue::Bool(false);
                program.runtime_input_wires.len()
            ]
            .into_boxed_slice();
            state.scalar_slots = vec![
                super::gpu_prepared_lowering::ScalarValue::Bool(false);
                program.scalar_slot_count
            ]
            .into();
            state.selection_results = vec![0usize; selection_len].into_boxed_slice();
            for (slot, value) in &program.scalar_initializers {
                if let Some(destination) = Arc::get_mut(&mut state.scalar_slots)
                    .expect("unpublished scalar slots")
                    .get_mut(*slot)
                {
                    destination.clone_from(value);
                }
            }
            for (input_index, wire) in program.runtime_input_wires.iter().enumerate() {
                if program.scalar_input_max_words.get(input_index).copied().unwrap_or(0) == 0 {
                    continue;
                }
                if program.device_scalar_wires.contains(wire) {
                    state.scalar_inputs[input_index] =
                        super::gpu_prepared_lowering::ScalarValue::Slot(input_index);
                    continue;
                }
                if let Some(slot) = program.scalar_slots.get(wire).copied() {
                    Arc::get_mut(&mut state.scalar_slots).expect("unpublished scalar slots")
                        [slot] =
                        super::gpu_prepared_lowering::ScalarValue::Int(num_bigint::BigInt::from(0));
                }
            }
            state.control_scratch =
                vec![super::gpu_prepared_lowering::ScalarValue::Bool(false); scratch_len]
                    .into_boxed_slice();
            state.control_results =
                vec![super::gpu_prepared_lowering::ScalarValue::Bool(false); result_len]
                    .into_boxed_slice();

            let mut control_by_node = BTreeMap::new();
            let mut control_index = 0;
            for node in &program.topology.nodes {
                if !matches!(
                    node.command.operation,
                    super::gpu_prepared_lowering::PreparedOperation::Gpu(_)
                ) {
                    control_by_node.insert(node.id, control_index);
                    control_index += 1;
                }
            }
            let mut steps = Vec::new();
            let mut native_used = vec![false; state.commands.len()];
            for replay in program.replay.iter() {
                append_replay_step(
                    replay,
                    &program,
                    &state.commands,
                    &mut native_used,
                    &control_by_node,
                    &mut steps,
                )?;
            }
            for command in state.commands.iter_mut() {
                if matches!(command.operation, PreparedOperation::Selection { .. }) {
                    command.selection_result =
                        control_by_node.get(&command.completion_event).copied();
                }
            }
            state.replay_steps = steps.into();
        }
        self.program = Some(program);
        Ok(self)
    }

    /// Compile the currently supported prepared graph shape without running
    /// the ordinary executor: one root RNS ModDown node with a transparent
    /// matrix output. The output owner is derived from the bound source
    /// context and one-prime level transition.
    pub fn from_rns_moddown_graph(
        graph: &ValidatedGraph,
        backend: &mut GpuDcrtBackend,
        source: &GpuFleetMatrix,
        program: &mut super::gpu_prepared_lowering::PreparedProgram,
    ) -> Result<Self, String> {
        let scope = graph.root_scope();
        let operation_nodes = scope
            .execution_order
            .iter()
            .filter(|node| !matches!(node.kind(), NodeKind::Input { .. }))
            .collect::<Vec<_>>();
        if operation_nodes.len() != 1 {
            return Err("prepared graph requires exactly one root operation".into());
        }
        let node = operation_nodes[0];
        let NodeKind::RnsModDown { plaintext_modulus, .. } = node.kind() else {
            return Err("prepared graph requires one RNS ModDown node".into());
        };
        let source_scope = graph.source.root_scope();
        let node_id = source_scope
            .node_id(node)
            .ok_or("prepared graph operation is not a root-scope node")?;
        let arguments = source_scope
            .arguments(node)
            .ok_or("prepared graph operation has an unresolved argument")?;
        if node.output_types().len() != 1 || arguments.len() != 1 {
            return Err("prepared graph requires one transparent output".into());
        }
        let input = arguments[0];
        let input_node = source_scope
            .node(input.node)
            .ok_or("prepared graph input is not declared in the root scope")?;
        if !matches!(input_node.kind(), NodeKind::Input { artifact: None, .. }) {
            return Err("prepared graph operation must consume a declared root input".into());
        }
        let output = WireRef { node: node_id, port: Port(0) };
        if source_scope.outputs() != [output] ||
            graph.source.outputs().len() != 1 ||
            graph.source.outputs().values().next().map(|root| root.value) != Some(output)
        {
            return Err("prepared graph requires exactly one exported ModDown root".into());
        }
        let plaintext_modulus = plaintext_modulus
            .evaluate(&graph.bindings)
            .map_err(|error| error.to_string())?
            .to_u64()
            .ok_or("prepared graph plaintext modulus is not u64")?;
        let output_type = scope
            .wire_types
            .get(&output)
            .and_then(|wire| wire.matrix_type())
            .ok_or("prepared graph output is not a matrix")?;
        let input_type = scope
            .wire_types
            .get(&input)
            .and_then(|wire| wire.matrix_type())
            .ok_or("prepared graph input is not a matrix")?;
        if input_type.rows != source.size().0 ||
            input_type.columns != source.size().1 ||
            output_type.rows != source.size().0 ||
            output_type.columns != source.size().1 ||
            source.shards().iter().any(|shard| {
                input_type.ring_dimension != shard.value.params().ring_dimension() as usize ||
                    input_type.modulus !=
                        num_bigint::BigInt::from(
                            shard.value.params().modulus().as_ref().clone(),
                        )
            })
        {
            return Err("prepared graph matrix shape does not match the bound fleet".into());
        }
        let target_parameters =
            backend.resource_parameters(output_type).map_err(|error| error.to_string())?;
        let target = source
            .shards()
            .iter()
            .map(|shard| {
                let parameters = target_parameters
                    .iter()
                    .find(|parameters| parameters.device_ids().contains(&shard.device_id))
                    .ok_or("prepared graph output has no matching device context")?;
                if output_type.modulus !=
                    num_bigint::BigInt::from(parameters.modulus().as_ref().clone())
                {
                    return Err("prepared graph output CRT basis differs from its target".into());
                }
                if shard.value.level() == 0 || parameters.crt_depth() == 0 {
                    return Err("prepared RNS ModDown source has no removable prime".into());
                }
                Ok(PreparedGpuTarget {
                    device: shard.device_id,
                    start: shard.global_column_start,
                    parameters: parameters.clone(),
                    rows: shard.value.row_size(),
                    columns: shard.value.col_size(),
                    level: parameters.crt_depth() - 1,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Self::new_rns_down(source, &target, plaintext_modulus, backend, program)
    }
    fn new_rns_down(
        source: &GpuFleetMatrix,
        target: &[PreparedGpuTarget],
        plaintext_modulus: u64,
        backend: &mut GpuDcrtBackend,
        program: &mut super::gpu_prepared_lowering::PreparedProgram,
    ) -> Result<Self, String> {
        if source.shards().len() != target.len() {
            return Err("prepared fleet conversion shape differs".into());
        }
        for (source_shard, target_shard) in source.shards().iter().zip(target) {
            if source_shard.device_id != target_shard.device ||
                source_shard.global_column_start != target_shard.start ||
                source_shard.value.level() != target_shard.level + 1 ||
                source_shard.value.row_size() != target_shard.rows ||
                source_shard.value.col_size() != target_shard.columns
            {
                return Err(
                    "prepared RNS ModDown requires matching placement and one-prime drop".into()
                );
            }
        }
        let mut descriptors =
            Vec::with_capacity(program.instance_count * source.shards().len() * 2);
        for instance_index in 0..program.instance_count {
            for (shard_index, (source_shard, target_shard)) in
                source.shards().iter().zip(target).enumerate()
            {
                let binding_base =
                    ((instance_index * source.shards().len() + shard_index) * 2) as u64;
                descriptors.push(PreparedMatrixDescriptor {
                    binding: prepared_binding_id(
                        binding_base,
                        source_shard.device_id,
                        instance_index,
                    ),
                    params: source_shard.value.params().clone(),
                    device: source_shard.device_id,
                    rows: source_shard.value.row_size(),
                    columns: source_shard.value.col_size(),
                    level: source_shard.value.level(),
                    is_ntt: false,
                });
                descriptors.push(PreparedMatrixDescriptor {
                    binding: prepared_binding_id(
                        binding_base + 1,
                        target_shard.device,
                        instance_index,
                    ),
                    params: target_shard.parameters.clone(),
                    device: target_shard.device,
                    rows: target_shard.rows,
                    columns: target_shard.columns,
                    level: target_shard.level,
                    is_ntt: false,
                });
            }
        }
        let (region, region_storages, binding_map) =
            reserve_prepared_matrices(backend, &descriptors)?;
        record_instance_storage_bindings(program, &descriptors, &binding_map);
        let make_instance =
            |instance_index: usize| -> Result<(Box<[PreparedCommand]>, Box<[usize]>), String> {
                let mut commands = Vec::with_capacity(source.shards().len() * 2);
                let mut outputs = Vec::with_capacity(source.shards().len());
                for (shard_index, (source_shard, target_shard)) in
                    source.shards().iter().zip(target).enumerate()
                {
                    let descriptor_index =
                        (instance_index * source.shards().len() + shard_index) * 2;
                    let source_descriptor = &descriptors[descriptor_index];
                    let target_descriptor = &descriptors[descriptor_index + 1];
                    let source_binding = binding_map
                        .get(&source_descriptor.binding)
                        .ok_or("missing region input binding")?;
                    let target_binding = binding_map
                        .get(&target_descriptor.binding)
                        .ok_or("missing region output binding")?;
                    let input_storage = region_storages
                        .get(&source_binding.storage)
                        .ok_or("missing region input storage")?
                        .clone();
                    let input_dispatch = input_storage
                        .reserve(std::slice::from_ref(&source_binding.request))?
                        .enter(Vec::new())?;
                    let mut input = GpuDCRTPolyMatrix::new_empty_with_state(
                        source_shard.value.params(),
                        source_shard.value.row_size(),
                        source_shard.value.col_size(),
                        source_shard.value.level(),
                        false,
                        None,
                    );
                    input.initialize_prepared_input(&source_shard.value)?;
                    drop(input_dispatch.finish()?);
                    let input = Arc::new(input);
                    let output_storage = region_storages
                        .get(&target_binding.storage)
                        .ok_or("missing region output storage")?
                        .clone();
                    let output_dispatch = output_storage
                        .reserve(std::slice::from_ref(&target_binding.request))?
                        .enter(Vec::new())?;
                    let matrix = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                        &target_shard.parameters,
                        target_shard.rows,
                        target_shard.columns,
                        target_shard.level,
                        false,
                        None,
                    ));
                    let plan = Arc::new(GpuPreparedModulusConversion::new_rns_down(
                        input.as_ref(),
                        matrix.as_ref(),
                        plaintext_modulus,
                    )?);
                    let transform = GpuPreparedTransform::new_forward(matrix.as_ref())?;
                    let command = GpuPreparedModulusConversion::bind(
                        Arc::clone(&plan),
                        Arc::clone(&input),
                        matrix.as_ref(),
                    )?;
                    drop(output_dispatch.finish()?);
                    commands.push(PreparedCommand::modulus(
                        command,
                        Arc::clone(&matrix),
                        target_shard.device,
                        target_shard.start,
                    ));
                    commands.push(PreparedCommand::transform(
                        transform,
                        matrix,
                        target_shard.device,
                        target_shard.start,
                    ));
                    outputs.push(commands.len() - 1);
                }
                Ok((commands.into_boxed_slice(), outputs.into_boxed_slice()))
            };
        let instances =
            (0..program.instance_count.max(1)).map(make_instance).collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_command_instances(instances, region, source.size().0, source.size().1))
    }

    pub fn run(&self) -> Result<PreparedGpuFleetOutput, PreparedGpuRunError> {
        self.run_with_inputs(&[])
    }

    pub(crate) fn run_with_inputs(
        &self,
        inputs: &[Arc<GpuDCRTPolyMatrix>],
    ) -> Result<PreparedGpuFleetOutput, PreparedGpuRunError> {
        let slot = self.pool.acquire()?;
        let instance = &self.pool.instances[slot];
        let mut state = instance.state.lock().expect("prepared GPU instance poisoned");
        // A native submit may enqueue work before reporting an error. Keep the
        // slot failed unless every shard has completed successfully.
        state.poisoned = true;
        let commands = &mut state.commands;
        for command in commands.iter_mut() {
            if let Err(error) = command.submit(inputs) {
                self.pool.poisoned.fetch_add(1, Ordering::Release);
                return Err(PreparedGpuRunError::Failed(error));
            }
        }
        state.poisoned = false;
        Ok(PreparedGpuFleetOutput {
            pool: Arc::clone(&self.pool),
            slot,
            rows: self.rows,
            columns: self.columns,
            output_descriptors: Arc::clone(&self.output_descriptors),
            output_names: Arc::clone(&self.output_names),
            output_indices: Arc::clone(&self.output_indices),
            scalar_output_commands: Arc::clone(&self.scalar_output_commands),
            host_output_descriptors: Arc::clone(&self.host_output_descriptors),
            scalar_slots: Some(Arc::clone(&state.scalar_slots)),
        })
    }

    /// Resolve the compatibility map into the immutable root order compiled
    /// during warmup. This adapter is outside the replay path; execution then
    /// receives positional roots and performs only fixed descriptor updates.
    pub(crate) fn ordered_runtime_inputs(
        &self,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<GpuDcrtBackend>>,
    ) -> Result<Box<[PreparedRuntimeValue]>, PreparedGpuRunError> {
        let program = self
            .program
            .as_ref()
            .ok_or_else(|| PreparedGpuRunError::Failed("prepared program is unavailable".into()))?;
        program
            .input_names
            .iter()
            .map(|(name, _)| {
                record_prepared_input_name_lookup();
                inputs
                    .get(name)
                    .ok_or_else(|| {
                        PreparedGpuRunError::Failed(format!("prepared input `{name}` is missing"))
                    })
                    .and_then(|value| {
                        super::fleet::prepared_runtime_value(value)
                            .map_err(|error| PreparedGpuRunError::Failed(error.to_string()))
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }

    /// Bind positional caller roots directly into the warmup-owned leaf
    /// slots. Root family expansion is represented by `runtime_input_roots`
    /// and `input_leaf_bindings`; replay performs no name or map lookup.
    pub(crate) fn run_with_runtime_roots(
        &self,
        roots: &[PreparedRuntimeValue],
    ) -> Result<PreparedGpuFleetOutput, PreparedGpuRunError> {
        let program = self
            .program
            .as_ref()
            .ok_or_else(|| PreparedGpuRunError::Failed("prepared program is unavailable".into()))?;
        if roots.len() != program.input_names.len() {
            return Err(PreparedGpuRunError::Failed(
                "prepared input root vector has the wrong number of values".into(),
            ));
        }
        let slot = self.pool.acquire()?;
        let instance = &self.pool.instances[slot];
        let mut state = instance.state.lock().expect("prepared GPU instance poisoned");
        state.poisoned = true;
        for (index, descriptor) in self.runtime_input_descriptors.iter().enumerate() {
            let root = roots.get(descriptor.root_index).ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared input root index is out of bounds".into())
            })?;
            let value = prepared_family_leaf(root, &descriptor.path)
                .map_err(PreparedGpuRunError::Failed)?
                .clone();
            if let (PreparedRuntimeValue::Int(integer), max_words) = (&value, descriptor.max_words)
            {
                let words =
                    usize::try_from(integer.bits().div_ceil(64)).unwrap_or(usize::MAX).max(1);
                if max_words != 0 && words > max_words {
                    return Err(PreparedGpuRunError::Failed(format!(
                        "prepared integer input exceeds warmup width: {words} words > {max_words}"
                    )));
                }
            }
            state.input_values[index].clone_from(&value);
            state.scalar_inputs[index] = match &value {
                PreparedRuntimeValue::Int(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Int(value.clone())
                }
                PreparedRuntimeValue::Real(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Real(*value)
                }
                PreparedRuntimeValue::Bool(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Bool(*value)
                }
                _ => super::gpu_prepared_lowering::ScalarValue::Bool(false),
            };
            if let Some(scalar_slot) = descriptor.scalar_slot {
                let scalar = state.scalar_inputs[index].clone();
                let slots = Arc::get_mut(&mut state.scalar_slots).expect("acquired scalar slots");
                let destination = slots.get_mut(scalar_slot).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared scalar destination is out of bounds".into(),
                    )
                })?;
                destination.clone_from(&scalar);
            }
        }
        let replay_steps = Arc::clone(&state.replay_steps);
        let input_values = state.input_values.as_ptr();
        let input_len = state.input_values.len();
        if let Err(error) = replay_nested_steps(
            self,
            &replay_steps,
            &mut state,
            // The input slots are immutable for the duration of replay; the
            // mutable state only updates command selection/output slots.
            unsafe { std::slice::from_raw_parts(input_values, input_len) },
            None,
        ) {
            self.pool.poisoned.fetch_add(1, Ordering::Release);
            return Err(error);
        }
        state.poisoned = false;
        Ok(PreparedGpuFleetOutput {
            pool: Arc::clone(&self.pool),
            slot,
            rows: self.rows,
            columns: self.columns,
            output_descriptors: Arc::clone(&self.output_descriptors),
            output_names: Arc::clone(&self.output_names),
            output_indices: Arc::clone(&self.output_indices),
            scalar_output_commands: Arc::clone(&self.scalar_output_commands),
            host_output_descriptors: Arc::clone(&self.host_output_descriptors),
            scalar_slots: Some(Arc::clone(&state.scalar_slots)),
        })
    }

    #[cfg(test)]
    pub(crate) fn run_with_runtime_inputs(
        &self,
        inputs: &[PreparedRuntimeValue],
    ) -> Result<PreparedGpuFleetOutput, PreparedGpuRunError> {
        let runtime_inputs = expand_prepared_runtime_inputs(
            self.program.as_ref().ok_or_else(|| {
                PreparedGpuRunError::Failed("prepared program is unavailable".into())
            })?,
            inputs,
        )
        .map_err(PreparedGpuRunError::Failed)?;
        for input in &runtime_inputs {
            match input {
                PreparedRuntimeValue::FleetSmallMatrix(value) => {
                    return Err(PreparedGpuRunError::Failed(format!(
                        "prepared compact input {}x{} has no bound SmallRhs command",
                        value.size().0,
                        value.size().1
                    )));
                }
                PreparedRuntimeValue::Bytes(_) => {}
                PreparedRuntimeValue::Family(value) => {
                    return Err(PreparedGpuRunError::Failed(format!(
                        "prepared family input of length {} has no bound family command",
                        value.len()
                    )));
                }
                PreparedRuntimeValue::Trapdoor { .. } |
                PreparedRuntimeValue::FleetMatrix(_) |
                PreparedRuntimeValue::Int(_) |
                PreparedRuntimeValue::Real(_) |
                PreparedRuntimeValue::Bool(_) => {}
            }
        }
        if runtime_inputs.len() !=
            self.pool.instances[0]
                .state
                .lock()
                .expect("prepared GPU instance poisoned")
                .input_values
                .len()
        {
            return Err(PreparedGpuRunError::Failed(
                "prepared input contract has the wrong number of values".into(),
            ));
        }
        for (index, input) in runtime_inputs.iter().enumerate() {
            let Some(&max_words) = self.scalar_input_max_words.get(index) else { continue };
            if max_words == 0 {
                continue;
            }
            if let PreparedRuntimeValue::Int(value) = input {
                let words = usize::try_from(value.bits().div_ceil(64)).unwrap_or(usize::MAX).max(1);
                if words > max_words {
                    return Err(PreparedGpuRunError::Failed(format!(
                        "prepared integer input exceeds warmup width: {words} words > {max_words}"
                    )));
                }
            }
        }
        let slot = self.pool.acquire()?;
        let instance = &self.pool.instances[slot];
        let mut state = instance.state.lock().expect("prepared GPU instance poisoned");
        state.poisoned = true;
        for (bound, input) in state.input_values.iter_mut().zip(&runtime_inputs) {
            if matches!(
                input,
                PreparedRuntimeValue::Trapdoor { .. } |
                    PreparedRuntimeValue::FleetMatrix(_) |
                    PreparedRuntimeValue::FleetSmallMatrix(_) |
                    PreparedRuntimeValue::Bytes(_) |
                    PreparedRuntimeValue::Family(_)
            ) {
                bound.clone_from(input);
            }
        }
        for (slot, input) in state.scalar_inputs.iter_mut().zip(&runtime_inputs) {
            if matches!(slot, super::gpu_prepared_lowering::ScalarValue::Slot(_)) {
                continue;
            }
            *slot = match input {
                PreparedRuntimeValue::Int(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Int(value.clone())
                }
                PreparedRuntimeValue::Real(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Real(*value)
                }
                PreparedRuntimeValue::Bool(value) => {
                    super::gpu_prepared_lowering::ScalarValue::Bool(*value)
                }
                _ => super::gpu_prepared_lowering::ScalarValue::Bool(false),
            };
        }
        let program = self.program.as_ref().expect("prepared program retained");
        let slots = Arc::get_mut(&mut state.scalar_slots).expect("acquired scalar slots");
        for (index, wire) in program.runtime_input_wires.iter().enumerate() {
            let Some(&slot) = program.scalar_slots.get(wire) else {
                continue;
            };
            let value = match runtime_inputs.get(index) {
                Some(PreparedRuntimeValue::Int(value)) => {
                    super::gpu_prepared_lowering::ScalarValue::Int(value.clone())
                }
                Some(PreparedRuntimeValue::Real(value)) => {
                    super::gpu_prepared_lowering::ScalarValue::Real(*value)
                }
                Some(PreparedRuntimeValue::Bool(value)) => {
                    super::gpu_prepared_lowering::ScalarValue::Bool(*value)
                }
                _ => continue,
            };
            slots[slot] = value;
        }
        let replay_steps = Arc::clone(&state.replay_steps);
        if let Err(error) =
            replay_nested_steps(self, &replay_steps, &mut state, &runtime_inputs, None)
        {
            self.pool.poisoned.fetch_add(1, Ordering::Release);
            return Err(error);
        }
        state.poisoned = false;
        Ok(PreparedGpuFleetOutput {
            pool: Arc::clone(&self.pool),
            slot,
            rows: self.rows,
            columns: self.columns,
            output_descriptors: Arc::clone(&self.output_descriptors),
            output_names: Arc::clone(&self.output_names),
            output_indices: Arc::clone(&self.output_indices),
            scalar_output_commands: Arc::clone(&self.scalar_output_commands),
            host_output_descriptors: Arc::clone(&self.host_output_descriptors),
            scalar_slots: Some(Arc::clone(&state.scalar_slots)),
        })
    }
}

fn replay_nested_steps(
    execution: &PreparedGpuFleetExecution,
    steps: &[super::gpu_prepared_control::PreparedExecutableCommand],
    state: &mut FleetInstanceState,
    inputs: &[PreparedRuntimeValue],
    parent_iteration: Option<usize>,
) -> Result<(), PreparedGpuRunError> {
    for step in steps {
        match step {
            super::gpu_prepared_control::PreparedExecutableCommand::Control(index) => {
                let command = execution.control_commands.get(*index).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared nested control index is out of bounds".into(),
                    )
                })?;
                let FleetInstanceState {
                    scalar_inputs,
                    scalar_slots,
                    control_scratch,
                    control_results,
                    selection_results,
                    ..
                } = state;
                super::gpu_prepared_control::execute_control_commands(
                    std::slice::from_ref(command),
                    scalar_inputs,
                    control_scratch,
                    &mut control_results[*index..=*index],
                    Arc::get_mut(scalar_slots).expect("acquired scalar slots"),
                    &mut selection_results[*index..=*index],
                    parent_iteration,
                )
                .map_err(PreparedGpuRunError::Failed)?;
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Native {
                index,
                variant,
                variant_indices,
            } => {
                let selected = parent_iteration
                    .and_then(|iteration| variant_indices.get(iteration).copied())
                    .unwrap_or(0);
                if *variant != selected {
                    continue;
                }
                let selected = state.commands[*index]
                    .selection_result
                    .map(|slot| state.selection_results[slot]);
                let command = state.commands.get_mut(*index).ok_or_else(|| {
                    PreparedGpuRunError::Failed(
                        "prepared nested native index is out of bounds".into(),
                    )
                })?;
                if let Some(selected) = selected {
                    command.set_selection(selected);
                }
                command.submit_runtime(inputs).map_err(PreparedGpuRunError::Failed)?;
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Subgraph { body } => {
                replay_nested_steps(execution, body, state, inputs, parent_iteration)?;
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Parallel { counts, waves } => {
                let active = parent_iteration
                    .and_then(|iteration| counts.get(iteration).copied())
                    .unwrap_or(usize::MAX);
                let mut iteration = 0;
                for wave in waves {
                    for body in wave {
                        if iteration >= active {
                            break;
                        }
                        replay_nested_steps(
                            execution,
                            std::slice::from_ref(body),
                            state,
                            inputs,
                            parent_iteration,
                        )?;
                        iteration += 1;
                    }
                    if iteration >= active {
                        break;
                    }
                }
            }
            super::gpu_prepared_control::PreparedExecutableCommand::Sequential {
                count,
                counts,
                offsets,
                banks,
                tail,
            } => {
                let active = parent_iteration
                    .and_then(|iteration| counts.get(iteration).copied())
                    .unwrap_or(*count);
                let base = parent_iteration
                    .and_then(|iteration| offsets.get(iteration).copied())
                    .unwrap_or_else(|| parent_iteration.unwrap_or(0).saturating_mul(*count));
                for iteration in 0..active {
                    let bank = iteration & 1;
                    replay_nested_steps(
                        execution,
                        &banks[bank],
                        state,
                        inputs,
                        Some(base + iteration),
                    )?;
                }
                if active % 2 == 1 {
                    replay_nested_steps(
                        execution,
                        tail,
                        state,
                        inputs,
                        Some(base + active.saturating_sub(1)),
                    )?;
                }
            }
        }
    }
    Ok(())
}

pub type PreparedGpuExecution = PreparedGpuFleetExecution;
pub type PreparedGpuOutput = PreparedGpuFleetOutput;

/// Build the owner-bearing tape directly from the lowered topology.  The
/// dispatcher is intentionally based on the fixed operation requirements, not
/// on ciphertext dimensions or graph names; adding another topology requires
/// adding its typed native command lowering rather than silently falling back
/// to the adaptive executor.
pub(crate) fn from_lowered_program(
    backend: &mut GpuDcrtBackend,
    inputs: &[PreparedRuntimeValue],
    program: &mut super::gpu_prepared_lowering::PreparedProgram,
) -> Result<PreparedGpuFleetExecution, String> {
    if program.instance_count == 0 || program.instance_count > usize::BITS as usize {
        return Err("prepared graph instance count exceeds the execution mask".into());
    }
    let inputs = expand_prepared_runtime_inputs(program, inputs)?;
    let has_matrix_input =
        inputs.iter().any(|input| matches!(input, PreparedRuntimeValue::FleetMatrix(_)));
    if !has_matrix_input &&
        program.topology.nodes.iter().any(|node| {
            matches!(
                node.command.operation,
                super::gpu_prepared_lowering::PreparedOperation::Gpu(
                    super::gpu_prepared_lowering::PreparedGpuOperation::HashCompactDecompose
                )
            )
        })
    {
        return from_bytes_only_prepared_program(backend, &inputs, program);
    }
    let mut matrix_inputs = Vec::new();
    let mut compact_inputs = BTreeMap::new();
    for (wire, input) in program.runtime_input_wires.iter().copied().zip(&inputs) {
        match input {
            PreparedRuntimeValue::FleetMatrix(value) |
            PreparedRuntimeValue::Trapdoor { public: value, .. } => {
                matrix_inputs.push(Arc::clone(value))
            }
            PreparedRuntimeValue::FleetSmallMatrix(value) => {
                compact_inputs.insert(wire, Arc::clone(value));
            }
            _ => {}
        }
    }
    if matrix_inputs.is_empty() &&
        !program.outputs.iter().any(|wire| program.wire_types[wire].matrix_type().is_some())
    {
        return Err("prepared byte-only graphs require the fixed compact-hash command path".into());
    }
    from_generic_matrix_program(backend, &matrix_inputs, &compact_inputs, &inputs, program)
}

fn from_bytes_only_prepared_program(
    backend: &mut GpuDcrtBackend,
    inputs: &[PreparedRuntimeValue],
    program: &mut super::gpu_prepared_lowering::PreparedProgram,
) -> Result<PreparedGpuFleetExecution, String> {
    use mxx_ir_core::node::{HashTagComponent, HashVariant};

    if backend.device_parameters().len() != 1 {
        return Err("prepared compact hash currently requires one configured GPU device".into());
    }
    let (node_id, hash_node) = program
        .node_sources
        .iter()
        .find(|(_, node)| matches!(node.kind, NodeKind::HashSample { .. }))
        .ok_or("prepared compact hash node missing")?;
    let NodeKind::HashSample { variant, tag_prefix, tag_components, digit_count, .. } =
        hash_node.kind()
    else {
        return Err("prepared compact hash node kind mismatch".into());
    };
    if !matches!(variant, HashVariant::Decomposed | HashVariant::SmallDecomposed) {
        return Err("prepared compact hash variant mismatch".into());
    }
    let output_wire = program.node_bindings[node_id].1[0];
    if program.outputs.as_ref() != [output_wire] {
        return Err("prepared compact hash requires one exported output".into());
    }
    let output_type =
        program.wire_types.get(&output_wire).ok_or("prepared compact hash output type missing")?;
    let (matrix_type, max_bound) = match output_type {
        mxx_ir_core::types::ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
            (matrix, max_coefficient_bound)
        }
        _ => return Err("prepared compact hash output is not a small matrix".into()),
    };
    let small = *variant == HashVariant::SmallDecomposed;
    let digits = digit_count
        .as_ref()
        .ok_or("prepared compact hash digit count missing")?
        .evaluate(&hash_node.environment)
        .map_err(|error| error.to_string())?
        .to_usize()
        .ok_or("prepared compact hash digit count is not usize")?;
    let params = backend
        .resource_parameters(matrix_type)
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or("prepared compact hash has no device parameters")?;
    let device = params.device_ids()[0];
    let compact_rows = matrix_type.rows;
    let scratch_rows = compact_rows / digits;
    let bound = max_bound.to_biguint().ok_or("prepared compact hash bound must be nonnegative")?;
    let key_index = program
        .runtime_input_wires
        .iter()
        .zip(inputs)
        .position(
            |(_, value)| matches!(value, PreparedRuntimeValue::Bytes(bytes) if bytes.len() == 32),
        )
        .ok_or("prepared compact hash key input missing")?;
    let arguments = &program.node_bindings[node_id].0;
    let mut tag = tag_prefix.clone();
    let mut operand_inputs = Vec::new();
    for component in tag_components {
        match component {
            HashTagComponent::Bytes(bytes) => {
                tag.push(0);
                tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                tag.extend_from_slice(bytes);
            }
            HashTagComponent::Integer(expression) => {
                let value = expression
                    .evaluate(&hash_node.environment)
                    .map_err(|error| error.to_string())?;
                tag.push(1);
                append_hash_tag_integer(&mut tag, &value);
            }
            HashTagComponent::Operand(index) => {
                let wire = *arguments.get(*index).ok_or("hash operand index out of range")?;
                let input = program
                    .inputs
                    .iter()
                    .position(|candidate| *candidate == wire)
                    .ok_or("hash operand is not a root input")?;
                operand_inputs.push(input);
            }
            HashTagComponent::Decimal(expression) => {
                let value = expression
                    .evaluate(&hash_node.environment)
                    .map_err(|error| error.to_string())?
                    .to_string();
                tag.push(2);
                tag.extend_from_slice(&(value.len() as u64).to_be_bytes());
                tag.extend_from_slice(value.as_bytes());
            }
            HashTagComponent::U64Le(expression) => {
                let value = expression
                    .evaluate(&hash_node.environment)
                    .map_err(|error| error.to_string())?
                    .to_u64()
                    .ok_or("hash tag value is not u64")?;
                tag.push(3);
                tag.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    let instance_count = program.instance_count.max(1);
    let mut commands = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut small_outputs = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let descriptors = (0..instance_count)
        .map(|instance| PreparedMatrixDescriptor {
            binding: prepared_binding_id(0x1000 + instance as u64, device, instance),
            params: params.clone(),
            device,
            rows: scratch_rows,
            columns: matrix_type.columns,
            level: params.crt_depth().saturating_sub(1),
            is_ntt: false,
        })
        .collect::<Vec<_>>();
    let compact_descriptors = (0..instance_count)
        .map(|_| PreparedCompactDescriptor {
            params: params.clone(),
            device,
            rows: compact_rows,
            columns: matrix_type.columns,
            bound: bound.clone(),
        })
        .collect::<Vec<_>>();
    let (mut region, storages, binding_map, compact_bindings) =
        reserve_prepared_resources(backend, &descriptors, &compact_descriptors)?;
    for instance in 0..instance_count {
        let scratch = allocate_prepared_matrix(
            &storages,
            *binding_map
                .get(&descriptors[instance].binding)
                .ok_or("compact hash scratch binding missing")?,
            &params,
            scratch_rows,
            matrix_type.columns,
            params.crt_depth().saturating_sub(1),
            false,
        )?;
        let compact = allocate_prepared_compact(
            &storages,
            compact_bindings[instance],
            &params,
            compact_rows,
            matrix_type.columns,
            bound.clone(),
        )?;
        let hash = GpuPreparedHashSample::bind(
            Arc::clone(&scratch),
            [0; 32],
            &tag,
            GpuMatrixSampleDist::Uniform,
            0.0,
            params.modulus().to_u64().unwrap_or(0).saturating_sub(1),
            matrix_type.columns,
            0,
            None,
        )?;
        let decompose = GpuPreparedCompactDecompose::bind(
            Arc::clone(&scratch),
            Arc::clone(&compact),
            small,
            Some(digits),
        )?;
        let mut hash_command = PreparedCommand::hash_sample(
            hash,
            key_index,
            operand_inputs.clone().into_boxed_slice(),
            tag.clone().into_boxed_slice(),
            scratch,
            device,
            0,
        );
        let topology = program
            .topology
            .nodes
            .iter()
            .find(|node| node.id == *node_id)
            .ok_or("prepared compact hash has no topology identity")?;
        hash_command.apply_topology(topology);
        commands[instance].push(hash_command);
        let compact_index = commands[instance].len();
        let mut compact_command = PreparedCommand::compact_decompose(decompose, compact, device, 0);
        compact_command.apply_topology(topology);
        commands[instance].push(compact_command);
        small_outputs[instance].push(compact_index);
    }
    provision_prepared_command_schedules(
        backend,
        &mut commands,
        &mut region,
        &program.topology.nodes,
    )?;
    let instances = commands
        .into_iter()
        .map(|commands| (commands.into_boxed_slice(), Vec::new().into_boxed_slice()))
        .collect::<Vec<_>>();
    let execution = PreparedGpuFleetExecution::from_command_instances(
        instances,
        region,
        compact_rows,
        matrix_type.columns,
    )
    .with_small_output_indices(
        small_outputs.into_iter().map(|indices| indices.into_boxed_slice()).collect(),
    );
    Ok(execution.with_program(Arc::new(program.clone()))?)
}

fn prepared_parameters_for_type(
    wire_type: &mxx_ir_core::types::ConcreteWireType,
    backend: &GpuDcrtBackend,
    device: i32,
) -> Result<GpuDCRTPolyParams, String> {
    let matrix_type =
        wire_type.matrix_type().ok_or("prepared conversion output is not a matrix")?;
    backend
        .resource_parameters(matrix_type)
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|parameters| parameters.device_ids().contains(&device))
        .ok_or_else(|| "prepared conversion output has no matching device parameters".into())
}

fn prepared_owner_for_wire(
    owners: &BTreeMap<WireRef, Arc<GpuDCRTPolyMatrix>>,
    aliases: &BTreeMap<WireRef, WireRef>,
    mut wire: WireRef,
) -> Option<Arc<GpuDCRTPolyMatrix>> {
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(wire) {
            return None;
        }
        if let Some(owner) = owners.get(&wire) {
            return Some(Arc::clone(owner));
        }
        wire = *aliases.get(&wire)?;
    }
}

fn prepared_operation_device(operation: &PreparedOperation) -> Option<i32> {
    match operation {
        PreparedOperation::Trapdoor { device, .. } |
        PreparedOperation::Preimage { device, .. } |
        PreparedOperation::ScalarOp { device, .. } |
        PreparedOperation::ScalarMatrixSelect { device, .. } |
        PreparedOperation::ScalarUpload { device, .. } |
        PreparedOperation::Threshold { device, .. } |
        PreparedOperation::ScalarPack { device, .. } |
        PreparedOperation::InputCopy { device, .. } |
        PreparedOperation::Arithmetic { device, .. } |
        PreparedOperation::Accumulate { device, .. } |
        PreparedOperation::Transform { device, .. } |
        PreparedOperation::Modulus { device, .. } |
        PreparedOperation::Transpose { device, .. } |
        PreparedOperation::ConcatRows { device, .. } |
        PreparedOperation::CenteredRebase { device, .. } |
        PreparedOperation::GadgetDecompose { device, .. } |
        PreparedOperation::Sampling { device, .. } |
        PreparedOperation::SmallRhs { device, .. } |
        PreparedOperation::HashSample { device, .. } |
        PreparedOperation::CompactDecompose { device, .. } |
        PreparedOperation::Reconstruction { device, .. } |
        PreparedOperation::Readback { device, .. } |
        PreparedOperation::Upload { device, .. } |
        PreparedOperation::CrtRecompose { device, .. } |
        PreparedOperation::Alias { device, .. } |
        PreparedOperation::Selection { device, .. } |
        PreparedOperation::LoopBody { device, .. } => Some(*device),
    }
}

fn prepared_schedule_for_operation(
    operation: &PreparedOperation,
) -> Result<Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSchedule>, String> {
    use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSchedulePlan;

    let schedule = match operation {
        PreparedOperation::Trapdoor { command, .. } => command.schedule()?,
        PreparedOperation::Preimage { command, .. } => command.schedule()?,
        PreparedOperation::ScalarOp { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarOp(command)], &[])?
        }
        PreparedOperation::ScalarMatrixSelect { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarMatrixSelect(command)], &[])?
        }
        PreparedOperation::ScalarUpload { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarUpload(command)], &[])?
        }
        PreparedOperation::Threshold { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Threshold(command)], &[])?
        }
        PreparedOperation::ScalarPack { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::ScalarPack(command)], &[])?
        }
        PreparedOperation::InputCopy { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::InputCopy(command)], &[])?
        }
        PreparedOperation::Arithmetic { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Arithmetic(command)], &[])?
        }
        PreparedOperation::Accumulate { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Accumulate(command)], &[])?
        }
        PreparedOperation::Transform { command, target, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Transform(command, target)], &[])?
        }
        PreparedOperation::Modulus { command, target, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Modulus(command, target)], &[])?
        }
        PreparedOperation::Transpose { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Transpose(command)], &[])?
        }
        PreparedOperation::ConcatRows { commands, .. } => {
            let plans = commands.iter().map(GpuPreparedSchedulePlan::InputCopy).collect::<Vec<_>>();
            GpuPreparedSchedule::new(&plans, &[])?
        }
        PreparedOperation::CenteredRebase { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::CenteredRebase(command)], &[])?
        }
        PreparedOperation::GadgetDecompose { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::GadgetDecompose(command)], &[])?
        }
        PreparedOperation::Sampling { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Sampling(command)], &[])?
        }
        PreparedOperation::SmallRhs { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::SmallRhs(command)], &[])?
        }
        PreparedOperation::HashSample { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::HashSample(command)], &[])?
        }
        PreparedOperation::CompactDecompose { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::CompactDecompose(command)], &[])?
        }
        PreparedOperation::Reconstruction { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Reconstruction(command)], &[])?
        }
        PreparedOperation::Readback { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Readback(command)], &[])?
        }
        PreparedOperation::Upload { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::Upload(command)], &[])?
        }
        PreparedOperation::CrtRecompose { command, .. } => {
            GpuPreparedSchedule::new(&[GpuPreparedSchedulePlan::CrtRecompose(command)], &[])?
        }
        PreparedOperation::Selection { candidates, .. } => {
            let plans = candidates
                .iter()
                .map(|candidate| GpuPreparedSchedulePlan::InputCopy(&candidate.command))
                .collect::<Vec<_>>();
            GpuPreparedSchedule::new(&plans, &[])?
        }
        PreparedOperation::Alias { .. } | PreparedOperation::LoopBody { .. } => return Ok(None),
    };
    Ok(Some(schedule))
}

fn provision_prepared_command_schedules(
    backend: &mut GpuDcrtBackend,
    instances: &mut [Vec<PreparedCommand>],
    region: &mut Arc<crate::gpu_memory::GpuMemoryRegion>,
    topology: &[super::gpu_prepared_lowering::PreparedTopologyNode],
) -> Result<(), String> {
    use super::fleet::PreparedScheduleStreamKey;

    let mut schedules = Vec::<Option<GpuPreparedSchedule>>::new();
    let mut schedule_streams = Vec::<Box<[PreparedScheduleStreamKey]>>::new();
    let mut schedule_indices =
        (0..instances.len()).map(|_| Vec::<Option<usize>>::new()).collect::<Vec<_>>();
    for instance in 0..instances.len() {
        for (command_index, command) in instances[instance].iter_mut().enumerate() {
            let schedule = prepared_schedule_for_operation(&command.operation)?;
            let Some(schedule) = schedule else {
                schedule_indices[instance].push(None);
                continue;
            };
            if command.completion_event == 0 {
                return Err("prepared native command has no factory-assigned identity".into());
            }
            let schedule_index = schedules.len();
            let keys = (0..schedule.stream_count())
                .map(|stream| PreparedScheduleStreamKey {
                    instance,
                    command: command_index,
                    stream,
                    device: prepared_operation_device(&command.operation)
                        .expect("GPU prepared command has a device"),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            schedules.push(Some(schedule));
            schedule_streams.push(keys);
            schedule_indices[instance].push(Some(schedule_index));
        }
    }

    let bindings = backend
        .provision_prepared_schedules_in_region(&mut schedules, &schedule_streams, region)
        .map_err(|error| error.to_string())?;
    drop(bindings.entries);

    for instance in 0..instances.len() {
        let mut command_schedules = Vec::<Option<Arc<GpuPreparedSchedule>>>::new();
        for command_index in 0..schedule_indices[instance].len() {
            let Some(schedule_index) = schedule_indices[instance][command_index] else {
                command_schedules.push(None);
                continue;
            };
            let schedule = schedules
                .get_mut(schedule_index)
                .and_then(Option::take)
                .ok_or("prepared schedule index disappeared")?;
            command_schedules.push(Some(Arc::new(schedule)));
        }
        // One IR node can launch several native stages and can own one shard
        // per device. Resolve exact identities only after every stage exists.
        let mut last_stage = BTreeMap::new();
        let mut preceding_stage = vec![None; command_schedules.len()];
        for (index, command) in instances[instance].iter().enumerate() {
            if command_schedules[index].is_some() {
                let device = prepared_operation_device(&command.operation)
                    .ok_or("prepared native command has no device")?;
                preceding_stage[index] =
                    last_stage.insert((command.completion_event, device, command.variant), index);
            }
        }
        let predecessors = instances[instance]
            .iter()
            .enumerate()
            .map(|(index, command)| {
                let mut predecessors = BTreeSet::new();
                if let Some(device) = prepared_operation_device(&command.operation) {
                    let mut pending = command.wait_events.to_vec();
                    let mut visited = BTreeSet::new();
                    while let Some(event) = pending.pop() {
                        if !visited.insert(event) {
                            continue;
                        }
                        let producers = last_stage
                            .range((event, device, 0)..=(event, device, usize::MAX))
                            .map(|(_, producer)| *producer)
                            .collect::<Vec<_>>();
                        if !producers.is_empty() {
                            predecessors.extend(producers);
                        } else if let Some(node) =
                            topology.iter().find(|node| node.completion == event)
                        {
                            // Scalar and alias nodes have no native event. Their
                            // native ancestors still carry the data dependency.
                            pending.extend(node.waits.iter().copied());
                        } else {
                            return Err("prepared dependency has no producer identity".to_owned());
                        }
                    }
                }
                predecessors.extend(preceding_stage[index]);
                Ok(predecessors)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut bound = command_schedules.iter().map(Option::is_none).collect::<Vec<_>>();
        for _ in 0..command_schedules.len() {
            let Some(command_index) = (0..command_schedules.len()).find(|index| {
                !bound[*index] && predecessors[*index].iter().all(|source| bound[*source])
            }) else {
                if bound.iter().any(|value| !value) {
                    return Err("prepared schedule dependency graph is not topological".into());
                }
                break;
            };
            let dependencies = predecessors[command_index]
                .iter()
                .map(|source| {
                    Arc::clone(
                        command_schedules[*source].as_ref().expect("prepared dependency schedule"),
                    )
                })
                .collect::<Vec<_>>();
            let schedule = command_schedules[command_index]
                .as_mut()
                .expect("prepared schedule index disappeared");
            Arc::get_mut(schedule)
                .expect("unpublished prepared schedule is shared")
                .bind_dependencies(&dependencies)?;
            instances[instance][command_index].attach_schedule(Arc::clone(schedule));
            bound[command_index] = true;
        }
    }
    Ok(())
}

/// Build the common owner-bearing tape for the small, non-fused matrix family.
/// The reservation and owner table are created before any native plan is
/// prepared; replay therefore only submits the already-bound commands.
fn from_generic_matrix_program(
    backend: &mut GpuDcrtBackend,
    inputs: &[Arc<GpuFleetMatrix>],
    compact_inputs: &BTreeMap<WireRef, Arc<GpuFleetSmallMatrix>>,
    runtime_inputs: &[PreparedRuntimeValue],
    program: &mut super::gpu_prepared_lowering::PreparedProgram,
) -> Result<PreparedGpuFleetExecution, String> {
    use super::gpu_prepared_lowering::{PreparedGpuOperation, PreparedOperation};

    let matrix_input_wires = program
        .runtime_input_wires
        .iter()
        .copied()
        .filter(|wire| !compact_inputs.contains_key(wire) && program.values.contains_key(wire))
        .collect::<Vec<_>>();
    let constant_matrix_wires = program
        .node_sources
        .iter()
        .filter_map(|(node_id, source)| {
            let NodeKind::ConstantMatrix { .. } = source.kind() else { return None };
            let wire = *program.node_bindings.get(node_id)?.1.first()?;
            let matrix = program.wire_types.get(&wire)?.matrix_type()?.clone();
            Some((*node_id, wire, matrix))
        })
        .collect::<Vec<_>>();
    if (!inputs.is_empty() && matrix_input_wires.len() != inputs.len()) ||
        (inputs.is_empty() && !matrix_input_wires.is_empty())
    {
        return Err("generic prepared matrix input contract mismatch".into());
    }
    let scalar_only = inputs.is_empty();
    let anchor_wire = program
        .outputs
        .iter()
        .copied()
        .find(|wire| program.wire_types[wire].matrix_type().is_some())
        .ok_or("prepared scalar-only graph has no matrix output descriptor")?;
    let anchor_type = program.wire_types[&anchor_wire]
        .matrix_type()
        .ok_or("prepared scalar-only graph output is not a matrix")?;
    let anchor_params = backend
        .resource_parameters(anchor_type)
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .ok_or("prepared scalar-only graph output has no device parameters")?;
    let anchor_device = anchor_params
        .device_ids()
        .first()
        .copied()
        .ok_or("prepared scalar-only graph output has no device")?;
    let shard_count = if scalar_only { 1 } else { inputs[0].shards().len() };
    if (!scalar_only && shard_count == 0) ||
        inputs.iter().any(|input| input.shards().len() != shard_count)
    {
        return Err("generic prepared matrix placement mismatch".into());
    }

    let mut gpu_nodes = Vec::new();
    let mut host_nodes = Vec::new();
    let mut aliases = BTreeMap::new();
    for node in &program.topology.nodes {
        match node.command.operation {
            PreparedOperation::Warmup | PreparedOperation::Scalar => {}
            PreparedOperation::ParallelLoop | PreparedOperation::SequentialLoop => {
                return Err("prepared loop/subgraph requires recursive scope instantiation".into());
            }
            PreparedOperation::Selection => {
                let output_wire = program
                    .node_bindings
                    .get(&node.id)
                    .and_then(|(_, outputs)| outputs.first())
                    .copied();
                if output_wire.is_some_and(|wire| program.family_members.contains_key(&wire)) {
                    continue;
                }
                let selection = program
                    .selection_commands
                    .get(&node.id)
                    .ok_or("generic selection has no lowered binding")?;
                if matches!(
                    selection,
                    super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { .. } |
                        super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { .. } |
                        super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { .. }
                ) {
                    continue;
                }
                let dynamic = !matches!(
                    selection,
                    super::gpu_prepared_lowering::PreparedSelection::Static { .. }
                );
                if dynamic {
                    gpu_nodes.push((node.id, PreparedGpuOperation::FixedCopies));
                    continue;
                }
                let candidate_location = match selection {
                    super::gpu_prepared_lowering::PreparedSelection::Static { location } => {
                        location
                    }
                    super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                        candidates, ..
                    } |
                    super::gpu_prepared_lowering::PreparedSelection::Select {
                        candidates, ..
                    } => candidates.first().ok_or("generic selection has no candidates")?,
                    super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { .. } |
                    super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { .. } |
                    super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { .. } => {
                        return Err("scalar selection cannot be a matrix command".into());
                    }
                };
                let (_, outputs) = program
                    .node_bindings
                    .get(&node.id)
                    .ok_or("generic selection node has no binding")?;
                let output = *outputs.first().ok_or("generic selection has no output")?;
                let source = program
                    .values
                    .iter()
                    .find(|(_, value)| value.owner == candidate_location.owner)
                    .map(|(wire, _)| *wire)
                    .ok_or("generic selection source is not bound")?;
                aliases.insert(output, source);
            }
            PreparedOperation::Alias => {
                let (arguments, outputs) = program
                    .node_bindings
                    .get(&node.id)
                    .ok_or("generic alias node has no binding")?;
                let source = *arguments.first().ok_or("generic alias has no source")?;
                let output = *outputs.first().ok_or("generic alias has no output")?;
                let source_location =
                    program.values.get(&source).ok_or("generic alias source has no location")?;
                let output_location =
                    program.values.get(&output).ok_or("generic alias output has no location")?;
                if source_location.owner != output_location.owner {
                    return Err("generic fixed view requires a copy command".into());
                }
                aliases.insert(output, source);
            }
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixBinary(_)) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::Transpose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ConcatRows) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::FixedCopies) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixMulAccumulate) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixMulSmallRhs) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixNegate) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::MatrixScale) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RingAutomorphism) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::Tensor) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ModulusSwitch) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ModulusReduce) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::CenteredExtend) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::BlockModSwitch) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsModUp) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsModDown) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::CenteredRebase) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::GadgetDecompose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::UniformResidueSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::UniformIntervalSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::GaussianSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::TrapdoorSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::PreimageSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::HashSample) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::HashCompactDecompose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::CrtRecompose) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsUpload) |
            PreparedOperation::Gpu(
                operation @ PreparedGpuOperation::PackPolynomialCoefficients,
            ) |
            PreparedOperation::Gpu(
                operation @ PreparedGpuOperation::LiftIntegerToConstantPolynomial,
            ) => {
                gpu_nodes.push((node.id, operation));
            }
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::RnsReadback) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ThresholdDecode) |
            PreparedOperation::Gpu(operation @ PreparedGpuOperation::ExtractCoefficient) => {
                host_nodes.push((node.id, operation));
            }
            PreparedOperation::Gpu(operation) => {
                return Err(format!(
                    "generic prepared matrix has no fixed command for {operation:?}"
                ));
            }
        }
    }
    if gpu_nodes.is_empty() && host_nodes.is_empty() {
        return Err("generic prepared matrix has no GPU operation".into());
    }
    let output_wire = *program.outputs.first().ok_or("generic matrix has no output")?;
    let output_shape = program.values.get(&output_wire).map_or_else(
        || (anchor_type.rows, anchor_type.columns),
        |output_location| {
            debug_assert_eq!(output_location.rows.start, 0);
            debug_assert_eq!(output_location.columns.start, 0);
            output_location.shape()
        },
    );

    // Reserve every staging/output owner in one transaction.  The key list
    // mirrors descriptor order and is consumed only while constructing the
    // fixed tape below.
    let mut descriptors = Vec::new();
    let mut descriptor_keys = Vec::new();
    let mut host_staging_descriptor_indices = BTreeMap::<(usize, usize, u32), usize>::new();
    let mut host_staging_specs = Vec::new();
    let mut compact_descriptors = Vec::new();
    let mut compact_descriptor_indices = BTreeMap::new();
    let instance_count = program.instance_count.max(1);
    for instance in 0..instance_count {
        for shard in 0..shard_count {
            let mut matrix_input_index = 0;
            for wire in matrix_input_wires.iter().copied() {
                let source = &inputs[matrix_input_index].shards()[shard];
                let location = program.values.get(&wire).ok_or("generic input has no location")?;
                if location.rows.start != 0 ||
                    location.columns.start != 0 ||
                    location.shape() != (source.value.row_size(), source.value.col_size()) ||
                    (location.format == super::gpu_prepared_lowering::PreparedFormat::Evaluation) !=
                        source.value.is_ntt()
                {
                    return Err("generic input owner contract mismatch".into());
                }
                if let Some(location) = program.values.get_mut(&wire) {
                    location.level = source.value.level();
                    location.device = source.device_id;
                }
                let binding =
                    prepared_binding_id(program.values[&wire].owner, source.device_id, instance);
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: source.value.params().clone(),
                    device: source.device_id,
                    rows: source.value.row_size(),
                    columns: source.value.col_size(),
                    level: source.value.level(),
                    is_ntt: source.value.is_ntt(),
                });
                descriptor_keys.push((instance, shard, wire));
                matrix_input_index += 1;
            }
            let source_device =
                if scalar_only { anchor_device } else { inputs[0].shards()[shard].device_id };
            for (_, wire, matrix) in &constant_matrix_wires {
                let output_params = prepared_parameters_for_type(
                    &mxx_ir_core::types::ConcreteWireType::Matrix(matrix.clone()),
                    backend,
                    source_device,
                )?;
                let binding =
                    prepared_binding_id(program.values[wire].owner, source_device, instance);
                if let Some(location) = program.values.get_mut(wire) {
                    location.level = output_params.moduli().len() - 1;
                    location.device = source_device;
                }
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: output_params.clone(),
                    device: source_device,
                    rows: matrix.rows,
                    columns: matrix.columns,
                    level: output_params.moduli().len() - 1,
                    is_ntt: program.values[wire].format ==
                        super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                });
                descriptor_keys.push((instance, shard, *wire));
            }
            for entry in &gpu_nodes {
                let node_id = entry.0;
                let operation = entry.1;
                let binding =
                    program.node_bindings.get(&node_id).ok_or("generic GPU node has no binding")?;
                let outputs = &binding.1;
                let wire = *outputs.first().ok_or("generic GPU node has no output")?;
                let node_source = &program.node_sources[&node_id];
                let output_type = node_source
                    .variant_output_types
                    .first()
                    .and_then(|types| types.first())
                    .unwrap_or(&program.wire_types[&wire]);
                let (rows, columns, format) = {
                    let location =
                        program.values.get(&wire).ok_or("generic output has no location")?;
                    let matrix =
                        output_type.matrix_type().ok_or("prepared output is not a matrix")?;
                    (0..matrix.rows, 0..matrix.columns, location.format)
                };
                let source_device =
                    if scalar_only { anchor_device } else { inputs[0].shards()[shard].device_id };
                if rows.start != 0 || columns.start != 0 {
                    return Err("generic output exceeds the input owner contract".into());
                }
                if matches!(operation, PreparedGpuOperation::PreimageSample) {
                    let mxx_ir_core::types::ConcreteWireType::Preimage {
                        matrix,
                        max_coefficient_bound,
                    } = output_type
                    else {
                        return Err("prepared preimage output is not relation typed".into());
                    };
                    let params = prepared_parameters_for_type(output_type, backend, source_device)?;
                    compact_descriptor_indices
                        .insert((instance, shard, node_id, 0), compact_descriptors.len());
                    compact_descriptors.push(PreparedCompactDescriptor {
                        params,
                        device: source_device,
                        rows: matrix.rows,
                        columns: matrix.columns,
                        bound: max_coefficient_bound
                            .to_biguint()
                            .ok_or("prepared preimage bound is negative")?,
                    });
                }
                if matches!(operation, PreparedGpuOperation::HashCompactDecompose) {
                    let (matrix_type, max_bound) = match output_type {
                        mxx_ir_core::types::ConcreteWireType::SmallMatrix {
                            matrix,
                            max_coefficient_bound,
                        } => (matrix, max_coefficient_bound),
                        _ => return Err("generic compact output is not a small matrix".into()),
                    };
                    let NodeKind::HashSample { digit_count, .. } =
                        node_source.variants.first().unwrap_or(&node_source.kind)
                    else {
                        return Err("generic compact hash node kind mismatch".into());
                    };
                    let digits = digit_count
                        .as_ref()
                        .ok_or("generic compact digit count missing")?
                        .evaluate(&program.node_sources[&node_id.to_owned()].environment)
                        .map_err(|error| error.to_string())?
                        .to_usize()
                        .ok_or("generic compact digit count is not usize")?;
                    let output_params =
                        prepared_parameters_for_type(output_type, backend, source_device)?;
                    let compact_rows = matrix_type.rows;
                    let bound = max_bound
                        .to_biguint()
                        .ok_or("generic compact bound must be nonnegative")?;
                    let binding =
                        prepared_binding_id(program.values[&wire].owner, source_device, instance);
                    descriptors.push(PreparedMatrixDescriptor {
                        binding,
                        params: output_params.clone(),
                        device: source_device,
                        rows: matrix_type.rows / digits,
                        columns: matrix_type.columns,
                        level: output_params.moduli().len() - 1,
                        is_ntt: false,
                    });
                    descriptor_keys.push((instance, shard, wire));
                    compact_descriptor_indices
                        .insert((instance, shard, node_id, 0), compact_descriptors.len());
                    compact_descriptors.push(PreparedCompactDescriptor {
                        params: output_params,
                        device: source_device,
                        rows: compact_rows,
                        columns: matrix_type.columns,
                        bound,
                    });
                    continue;
                }
                if matches!(operation, PreparedGpuOperation::GadgetDecompose) {
                    let mxx_ir_core::types::ConcreteWireType::Preimage {
                        matrix,
                        max_coefficient_bound,
                    } = output_type
                    else {
                        return Err("generic gadget output is not compact typed".into());
                    };
                    let bound = max_coefficient_bound
                        .to_biguint()
                        .ok_or("generic gadget bound must be nonnegative")?;
                    let compact_index = compact_descriptors.len();
                    compact_descriptor_indices.insert((instance, shard, node_id, 0), compact_index);
                    compact_descriptors.push(PreparedCompactDescriptor {
                        params: prepared_parameters_for_type(output_type, backend, source_device)?,
                        device: source_device,
                        rows: matrix.rows,
                        columns: matrix.columns,
                        bound,
                    });
                }
                let output_params =
                    prepared_parameters_for_type(output_type, backend, source_device)?;
                let binding =
                    prepared_binding_id(program.values[&wire].owner, source_device, instance);
                let output_level = output_params.moduli().len() - 1;
                if let Some(location) = program.values.get_mut(&wire) {
                    location.level = output_level;
                    location.device = source_device;
                }
                descriptors.push(PreparedMatrixDescriptor {
                    binding,
                    params: output_params,
                    device: source_device,
                    rows: rows.end,
                    columns: columns.end,
                    level: output_level,
                    is_ntt: format == super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                });
                descriptor_keys.push((instance, shard, wire));
            }
            for (node_id, operation) in &host_nodes {
                if matches!(operation, PreparedGpuOperation::ThresholdDecode) {
                    continue;
                }
                let source_wire = *program
                    .node_bindings
                    .get(node_id)
                    .and_then(|(arguments, _)| arguments.first())
                    .ok_or("generic host source is missing")?;
                let source_location = program
                    .values
                    .get(&source_wire)
                    .ok_or("generic host source has no location")?;
                if source_location.format !=
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation
                {
                    continue;
                }
                let source_matrix = program
                    .wire_types
                    .get(&source_wire)
                    .and_then(|ty| ty.matrix_type())
                    .ok_or("generic host source is not a matrix")?;
                let params = backend
                    .resource_parameters(source_matrix)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .find(|parameters| parameters.device_ids().contains(&source_device))
                    .ok_or("generic host source has no device parameters")?;
                let binding_owner = program
                    .values
                    .values()
                    .map(|value| value.owner)
                    .max()
                    .unwrap_or(0)
                    .checked_add(1)
                    .and_then(|owner| owner.checked_add(*node_id as u64))
                    .ok_or("generic host staging owner overflow")?;
                let binding = prepared_binding_id(binding_owner, source_device, instance);
                host_staging_specs.push((
                    instance,
                    shard,
                    *node_id,
                    PreparedMatrixDescriptor {
                        binding,
                        params: params.clone(),
                        device: source_device,
                        rows: source_matrix.rows,
                        columns: source_matrix.columns,
                        level: params.moduli().len() - 1,
                        is_ntt: false,
                    },
                ));
            }
        }
    }
    for (instance, shard, node_id, descriptor) in host_staging_specs {
        let index = descriptors.len();
        descriptors.push(descriptor);
        host_staging_descriptor_indices.insert((instance, shard, node_id), index);
    }
    let descriptor_indices = descriptor_keys
        .iter()
        .copied()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<BTreeMap<_, _>>();
    let mut variant_descriptor_indices = BTreeMap::new();
    let mut typed_descriptor_indices = BTreeMap::new();
    for (index, &(instance, shard, wire)) in descriptor_keys.iter().enumerate() {
        if let Some(matrix) = program.wire_types[&wire].matrix_type() {
            typed_descriptor_indices.insert((instance, shard, wire, matrix.clone()), index);
        }
    }
    for instance in 0..instance_count {
        for shard in 0..shard_count {
            for (node, operation) in &gpu_nodes {
                let wire = program.node_bindings[node].1[0];
                let base = &descriptors[descriptor_indices[&(instance, shard, wire)]];
                let source = &program.node_sources[node];
                let base = base.clone();
                for (variant, types) in source.variant_output_types.iter().enumerate() {
                    let matrix =
                        types[0].matrix_type().ok_or("prepared variant is not a matrix")?;
                    let params = backend
                        .resource_parameters(matrix)
                        .map_err(|error| error.to_string())?
                        .into_iter()
                        .find(|parameters| parameters.device_ids().contains(&base.device))
                        .ok_or("prepared variant has no device parameters")?;
                    let index = descriptors.len();
                    let mut scratch_rows = matrix.rows;
                    if matches!(operation, PreparedGpuOperation::PreimageSample) {
                        let mxx_ir_core::types::ConcreteWireType::Preimage {
                            max_coefficient_bound,
                            ..
                        } = &types[0]
                        else {
                            return Err("prepared preimage variant is not relation typed".into());
                        };
                        let bound = max_coefficient_bound
                            .to_biguint()
                            .ok_or("prepared preimage bound is negative")?;
                        let existing = compact_descriptor_indices
                            .iter()
                            .filter(|((i, s, n, _), _)| (*i, *s, *n) == (instance, shard, *node))
                            .map(|(_, index)| *index)
                            .find(|index| {
                                let entry = &compact_descriptors[*index];
                                entry.params.context_identity() == params.context_identity() &&
                                    entry.rows == matrix.rows &&
                                    entry.columns == matrix.columns &&
                                    entry.bound == bound
                            });
                        let compact_index = existing.unwrap_or_else(|| {
                            let index = compact_descriptors.len();
                            compact_descriptors.push(PreparedCompactDescriptor {
                                params: params.clone(),
                                device: base.device,
                                rows: matrix.rows,
                                columns: matrix.columns,
                                bound,
                            });
                            index
                        });
                        compact_descriptor_indices
                            .insert((instance, shard, *node, variant), compact_index);
                    }
                    if matches!(operation, PreparedGpuOperation::HashCompactDecompose) {
                        let NodeKind::HashSample { digit_count, .. } = &source.variants[variant]
                        else {
                            return Err("prepared compact variant is not a hash".into());
                        };
                        let digits = digit_count
                            .as_ref()
                            .ok_or("prepared compact digit count missing")?
                            .evaluate(&source.environment)
                            .map_err(|error| error.to_string())?
                            .to_usize()
                            .ok_or("prepared compact digit count is not usize")?;
                        scratch_rows /= digits;
                        let mxx_ir_core::types::ConcreteWireType::SmallMatrix {
                            max_coefficient_bound,
                            ..
                        } = &types[0]
                        else {
                            return Err("prepared compact variant output is not compact".into());
                        };
                        let bound = max_coefficient_bound
                            .to_biguint()
                            .ok_or("prepared compact bound is negative")?;
                        let rows = matrix.rows;
                        let existing = compact_descriptor_indices
                            .iter()
                            .filter(|((owner_instance, owner_shard, owner_node, _), _)| {
                                (*owner_instance, *owner_shard, *owner_node) ==
                                    (instance, shard, *node)
                            })
                            .map(|(_, index)| *index)
                            .find(|index| {
                                let entry = &compact_descriptors[*index];
                                entry.params.context_identity() == params.context_identity() &&
                                    entry.rows == rows &&
                                    entry.columns == matrix.columns &&
                                    entry.bound == bound
                            });
                        let compact_index = existing.unwrap_or_else(|| {
                            let index = compact_descriptors.len();
                            compact_descriptors.push(PreparedCompactDescriptor {
                                params: params.clone(),
                                device: base.device,
                                rows,
                                columns: matrix.columns,
                                bound,
                            });
                            index
                        });
                        compact_descriptor_indices
                            .insert((instance, shard, *node, variant), compact_index);
                    }
                    descriptors.push(PreparedMatrixDescriptor {
                        rows: scratch_rows,
                        columns: matrix.columns,
                        level: params.moduli().len() - 1,
                        params,
                        ..base.clone()
                    });
                    variant_descriptor_indices.insert((instance, shard, *node, variant), index);
                    typed_descriptor_indices.insert((instance, shard, wire, matrix.clone()), index);
                }
            }
        }
    }
    // A liveness color may have several finite CRT classes. Each class receives
    // its own accepted slot; headers may only vary shape within that class.
    let mut classes = BTreeMap::new();
    let mut base_classes = BTreeSet::new();
    let mut next_owner = descriptors
        .iter()
        .map(|descriptor| descriptor.binding.owner)
        .chain(program.values.values().map(|value| value.owner))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("prepared owner id overflow")?;
    for descriptor in &mut descriptors {
        let class = (descriptor.binding.owner, descriptor.params.context_identity());
        descriptor.binding.owner = *classes.entry(class).or_insert_with(|| {
            if base_classes.insert((descriptor.binding.owner, descriptor.device)) {
                return descriptor.binding.owner;
            }
            let owner = next_owner;
            next_owner += 1;
            owner
        });
    }
    let mut physical_descriptors = BTreeMap::<_, PreparedMatrixDescriptor>::new();
    for descriptor in &descriptors {
        physical_descriptors
            .entry(descriptor.binding)
            .and_modify(|capacity| {
                capacity.rows = capacity.rows.max(descriptor.rows);
                capacity.columns = capacity.columns.max(descriptor.columns);
                capacity.level = capacity.level.max(descriptor.level);
            })
            .or_insert_with(|| descriptor.clone());
    }
    let physical_descriptors = physical_descriptors.into_values().collect::<Vec<_>>();
    let (mut region, region_storages, binding_map, compact_bindings) =
        reserve_prepared_resources(backend, &physical_descriptors, &compact_descriptors)?;
    record_instance_storage_bindings(program, &descriptors, &binding_map);
    let mut instances = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut output_indices = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut small_output_indices = (0..instance_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut allocated_compact = BTreeMap::new();
    let mut allocated = BTreeMap::new();
    let mut headers = BTreeMap::new();
    // Replicas of one logical sampler advance identical streams on all devices.
    let mut sampler_rngs = BTreeMap::<(usize, u32, usize), rand::rngs::StdRng>::new();
    let physical_layouts = physical_descriptors
        .iter()
        .map(|descriptor| (binding_map[&descriptor.binding].identity, descriptor))
        .collect::<BTreeMap<_, _>>();
    let mut allocate_prepared_matrix = |storages: &BTreeMap<u64, Arc<GpuPreparedStorage>>,
                                        binding: PreparedMatrixBinding,
                                        params: &GpuDCRTPolyParams,
                                        rows: usize,
                                        columns: usize,
                                        level: usize,
                                        is_ntt: bool| {
        let shape = (binding.identity, rows, columns, level, is_ntt);
        if let Some(owner) = headers.get(&shape) {
            return Ok(Arc::clone(owner));
        }
        let backing = if let Some(owner) = allocated.get(&binding.identity) {
            Arc::clone(owner)
        } else {
            let layout = physical_layouts[&binding.identity];
            let owner = allocate_prepared_matrix(
                storages,
                binding,
                &layout.params,
                layout.rows,
                layout.columns,
                layout.level,
                layout.is_ntt,
            )?;
            allocated.insert(binding.identity, Arc::clone(&owner));
            owner
        };
        if params.context_identity() != backing.params().context_identity() {
            return Err("prepared header has a different CRT context than its physical owner".into());
        }
        let owner = if (rows, columns, level, is_ntt) ==
            (backing.row_size(), backing.col_size(), backing.level(), backing.is_ntt())
        {
            backing
        } else {
            GpuDCRTPolyMatrix::prepared_shape(backing, rows, columns, level, is_ntt)?
        };
        headers.insert(shape, Arc::clone(&owner));
        Ok::<_, String>(owner)
    };

    for instance in 0..instance_count {
        for shard in 0..shard_count {
            let source = if scalar_only {
                let descriptor_index = descriptor_indices
                    .get(&(instance, shard, anchor_wire))
                    .copied()
                    .ok_or("prepared scalar-only anchor descriptor is missing")?;
                let descriptor = &descriptors[descriptor_index];
                let value = allocate_prepared_matrix(
                    &region_storages,
                    *binding_map
                        .get(&descriptor.binding)
                        .ok_or("prepared scalar-only anchor binding is missing")?,
                    &descriptor.params,
                    descriptor.rows,
                    descriptor.columns,
                    descriptor.level,
                    descriptor.is_ntt,
                )?;
                GpuColumnShard { device_id: anchor_device, global_column_start: 0, value }
            } else {
                inputs[0].shards()[shard].clone()
            };
            let shard_output_begin = output_indices[instance].len();
            let mut owners = BTreeMap::<WireRef, Arc<GpuDCRTPolyMatrix>>::new();
            let mut secrets =
                BTreeMap::<WireRef, (Arc<GpuDCRTTrapdoor>, Option<(usize, usize)>)>::new();
            for (node_id, wire, matrix) in &constant_matrix_wires {
                let descriptor = &descriptors[descriptor_indices[&(instance, shard, *wire)]];
                let binding = *binding_map
                    .get(&descriptor.binding)
                    .ok_or("prepared constant binding is missing")?;
                let mut owner = crate::backend::poly_gpu::gpu_prepared::allocate_prepared_matrix(
                    &region_storages,
                    binding,
                    &descriptor.params,
                    matrix.rows,
                    matrix.columns,
                    descriptor.level,
                    descriptor.is_ntt,
                )?;
                let constant = match program.node_sources[node_id].kind() {
                    NodeKind::ConstantMatrix { value, .. } => match value {
                        mxx_ir_core::node::ConstantMatrix::Zero => {
                            GpuMatrixRangeConstant::Zero { total_columns: matrix.columns }
                        }
                        mxx_ir_core::node::ConstantMatrix::Identity => {
                            GpuMatrixRangeConstant::Identity
                        }
                        mxx_ir_core::node::ConstantMatrix::UnitRow { index } => {
                            GpuMatrixRangeConstant::UnitRow {
                                total_columns: matrix.columns,
                                index: index
                                    .evaluate(&program.node_sources[node_id].environment)
                                    .map_err(|error| error.to_string())?
                                    .to_usize()
                                    .ok_or("prepared constant unit-row index is invalid")?,
                            }
                        }
                        mxx_ir_core::node::ConstantMatrix::UnitColumn { index } => {
                            GpuMatrixRangeConstant::UnitColumn {
                                index: index
                                    .evaluate(&program.node_sources[node_id].environment)
                                    .map_err(|error| error.to_string())?
                                    .to_usize()
                                    .ok_or("prepared constant unit-column index is invalid")?,
                            }
                        }
                        mxx_ir_core::node::ConstantMatrix::Gadget { base, small } => {
                            let base = base
                                .evaluate(&program.node_sources[node_id].environment)
                                .map_err(|error| error.to_string())?;
                            let expected = num_bigint::BigInt::from(1u8) <<
                                descriptor.params.base_bits() as usize;
                            if base != expected {
                                return Err(
                                    "prepared constant gadget base differs from device base".into(),
                                );
                            }
                            GpuMatrixRangeConstant::Gadget { small: *small, digit_count: None }
                        }
                        _ => {
                            return Err(
                                "prepared constant matrix kind has no fixed GPU fill descriptor"
                                    .into(),
                            )
                        }
                    },
                    _ => return Err("prepared constant source kind changed during warmup".into()),
                };
                Arc::get_mut(&mut owner)
                    .ok_or("prepared constant owner was published before initialization")?
                    .fill_constant_columns(
                        0..matrix.rows,
                        0..matrix.columns,
                        source.global_column_start,
                        constant,
                    )?;
                owners.insert(*wire, owner);
            }
            for (index, wire) in program.runtime_input_wires.iter().enumerate() {
                if let PreparedRuntimeValue::Trapdoor { secret, .. } = &runtime_inputs[index] {
                    let replica = secret
                        .values
                        .iter()
                        .position(|value| value.r.params().device_ids().contains(&source.device_id))
                        .ok_or("prepared trapdoor device replica is unavailable")?;
                    secrets.insert(
                        *wire,
                        (Arc::clone(&secret.values[replica]), Some((index, replica))),
                    );
                }
            }
            let mut device_scalars = BTreeMap::new();
            let mut compact_owners = BTreeMap::<WireRef, Arc<GpuSmallMatrix>>::new();
            let mut compact_typed_owners = BTreeMap::new();
            let mut matrix_input_index = 0;
            for wire in matrix_input_wires.iter().copied() {
                let descriptor = &descriptors[descriptor_indices[&(instance, shard, wire)]];
                let source = &inputs[matrix_input_index].shards()[shard];
                let staged = allocate_prepared_matrix(
                    &region_storages,
                    *binding_map.get(&descriptor.binding).ok_or("generic input binding missing")?,
                    source.value.params(),
                    source.value.row_size(),
                    source.value.col_size(),
                    source.value.level(),
                    source.value.is_ntt(),
                )?;
                let copy = GpuPreparedInputCopy::bind(
                    Arc::clone(&staged),
                    Arc::clone(&source.value),
                    None,
                )?;
                instances[instance].push(PreparedCommand::input_copy(
                    copy,
                    matrix_input_index * shard_count + shard,
                    Arc::clone(&staged),
                    source.device_id,
                    source.global_column_start,
                ));
                let topology_root = program
                    .input_leaf_bindings
                    .get(&wire)
                    .map(|binding| binding.root)
                    .unwrap_or(wire);
                let topology = program
                    .topology
                    .nodes
                    .iter()
                    .find(|node| program.node_bindings[&node.id].1.contains(&topology_root))
                    .ok_or("prepared input has no topology identity")?;
                instances[instance]
                    .last_mut()
                    .expect("prepared input command")
                    .apply_topology(topology);
                if program.outputs.contains(&wire) {
                    output_indices[instance].push(instances[instance].len() - 1);
                }
                owners.insert(wire, staged);
                matrix_input_index += 1;
            }

            for (node_id, operation) in &host_nodes {
                if !matches!(operation, PreparedGpuOperation::ThresholdDecode) {
                    continue;
                }
                let node_source = &program.node_sources[node_id];
                let kinds = if node_source.variants.is_empty() {
                    std::slice::from_ref(&node_source.kind)
                } else {
                    &node_source.variants
                };
                let (arguments, outputs) = &program.node_bindings[node_id];
                let mut source_wire = arguments[0];
                while let Some(parent) = aliases.get(&source_wire) {
                    source_wire = *parent;
                }
                let mut records = Vec::new();
                let mut maximum_words = 1;
                let mut maximum_count = 1;
                for (variant, kind) in kinds.iter().enumerate() {
                    let NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } = kind
                    else {
                        unreachable!()
                    };
                    let plaintext = plaintext_modulus
                        .evaluate(&node_source.environment)
                        .map_err(|error| error.to_string())?
                        .to_biguint()
                        .ok_or("negative plaintext")?;
                    let count = length
                        .evaluate(&node_source.environment)
                        .map_err(|error| error.to_string())?
                        .to_usize()
                        .ok_or("invalid threshold length")?;
                    let descriptor_index = node_source
                        .variant_input_types
                        .get(variant)
                        .and_then(|types| types.first())
                        .and_then(|ty| ty.matrix_type())
                        .and_then(|matrix| {
                            typed_descriptor_indices.get(&(
                                instance,
                                shard,
                                source_wire,
                                matrix.clone(),
                            ))
                        })
                        .copied()
                        .unwrap_or(descriptor_indices[&(instance, shard, source_wire)]);
                    let descriptor = &descriptors[descriptor_index];
                    let source = allocate_prepared_matrix(
                        &region_storages,
                        binding_map[&descriptor.binding],
                        &descriptor.params,
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.level,
                        descriptor.is_ntt,
                    )?;
                    maximum_words = maximum_words.max(if *output_bool {
                        1
                    } else {
                        plaintext.iter_u64_digits().len() + 1
                    });
                    maximum_count = maximum_count.max(count);
                    records.push((source, plaintext, count, *output_bool, descriptor.device));
                }
                let shared_output = gpu_prepared_scalar::allocate_scalar_buffer(
                    backend,
                    &mut region,
                    &records[0].0,
                    records[0].4,
                    maximum_count,
                    maximum_words,
                )?;
                let topology = program
                    .topology
                    .nodes
                    .iter()
                    .find(|node| node.id == *node_id)
                    .ok_or("threshold topology missing")?;
                for (variant, (source, plaintext, count, output_bool, device)) in
                    records.into_iter().enumerate()
                {
                    let params = source.params().clone();
                    let (_, workspace) =
                        GpuPreparedThreshold::layout(&source, &plaintext, count, output_bool)?;
                    let claims = [
                        GpuTracedClaim::matrix(1, 1, source.level(), false),
                        GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                            kind: GpuPreparedSlotKind::BatchWorkspace,
                            bytes: workspace,
                            alignment: 8,
                        }),
                    ];
                    let (staging, threshold) = bind_prepared_claims(
                        backend,
                        &mut region,
                        &params,
                        device,
                        &claims,
                        || {
                            let staging = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
                                &params,
                                1,
                                1,
                                source.level(),
                                false,
                                None,
                            ));
                            let threshold = GpuPreparedThreshold::bind(
                                Arc::clone(&staging),
                                &plaintext,
                                count,
                                output_bool,
                                Some(Arc::clone(&shared_output)),
                            )?;
                            Ok((staging, threshold))
                        },
                    )?;
                    let first = instances[instance].len();
                    let copy = GpuPreparedInputCopy::bind(
                        Arc::clone(&staging),
                        Arc::clone(&source),
                        None,
                    )?;
                    instances[instance].push(PreparedCommand::input_copy_from_owner(
                        copy,
                        Arc::clone(&source),
                        Arc::clone(&staging),
                        device,
                        0,
                    ));
                    if source.is_ntt() {
                        let inverse = GpuPreparedTransform::new_inverse(&staging)?;
                        instances[instance]
                            .push(PreparedCommand::transform(inverse, staging, device, 0));
                    }
                    instances[instance].push(PreparedCommand::new(
                        crate::backend::poly_gpu::gpu_prepared::PreparedOperation::Threshold {
                            command: threshold,
                            device,
                            node: *node_id,
                            output_bool,
                        },
                    ));
                    for command in &mut instances[instance][first..] {
                        command.apply_topology(topology);
                        command.variant = variant;
                    }
                }
                for (index, wire) in outputs.iter().enumerate() {
                    device_scalars.insert(*wire, (Arc::clone(&shared_output), index));
                }
            }

            prepare_scalar_commands(
                backend,
                &mut region,
                program,
                runtime_inputs,
                &source.value,
                source.device_id,
                &mut device_scalars,
                &mut instances[instance],
            )?;
            for (node_id, operation) in &gpu_nodes {
                let node_source = program.node_sources[node_id].clone();
                let output_wire = program.node_bindings[node_id].1[0];
                let variant_kinds = if node_source.variants.is_empty() {
                    std::slice::from_ref(&node_source.kind)
                } else {
                    &node_source.variants
                };
                for (variant, kind) in variant_kinds.iter().enumerate() {
                    let descriptor = &descriptors[*variant_descriptor_indices
                        .get(&(instance, shard, *node_id, variant))
                        .unwrap_or(&descriptor_indices[&(instance, shard, output_wire)])];
                    let input_types = node_source.variant_input_types.get(variant);
                    let output_types = node_source.variant_output_types.get(variant);
                    let node_source = super::gpu_prepared_lowering::PreparedNodeSource {
                        kind: kind.clone(),
                        environment: node_source.environment.clone(),
                        variants: Box::new([]),
                        variant_indices: Box::new([]),
                        variant_input_types: Box::new([]),
                        variant_output_types: Box::new([]),
                    };
                    let first_command = instances[instance].len();
                    let previous_outputs = output_indices[instance].len();
                    let previous_small_outputs = small_output_indices[instance].len();
                    let result = (|| -> Result<(), String> {
                        let (arguments, outputs) = program
                            .node_bindings
                            .get(node_id)
                            .ok_or("generic node binding missing")?;
                        let output_wire = *outputs.first().ok_or("generic output wire missing")?;
                        let mut location = program
                            .values
                            .get(&output_wire)
                            .ok_or("generic output location missing")?
                            .clone();
                        if let Some(matrix) = output_types
                            .and_then(|types| types.first())
                            .and_then(|ty| ty.matrix_type())
                        {
                            location.rows = 0..matrix.rows;
                            location.columns = 0..matrix.columns;
                        }
                        if let Some(types) = input_types {
                            for (wire, ty) in arguments.iter().zip(types.iter()) {
                                if matches!(
                                    ty,
                                    mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }
                                ) {
                                    continue;
                                }
                                if matches!(
                                    ty,
                                    mxx_ir_core::types::ConcreteWireType::SmallMatrix { .. } |
                                        mxx_ir_core::types::ConcreteWireType::Preimage { .. }
                                ) {
                                    if let Some(owner) =
                                        compact_typed_owners.get(&(*wire, ty.clone()))
                                    {
                                        compact_owners.insert(*wire, Arc::clone(owner));
                                    } else if !compact_inputs.contains_key(wire) {
                                        return Err(
                                            "prepared compact source variant is unavailable".into(),
                                        );
                                    }
                                    continue;
                                }
                                let Some(matrix) = ty.matrix_type() else { continue };
                                let mut producer = *wire;
                                while let Some(source) = aliases.get(&producer) {
                                    producer = *source;
                                }
                                let source_descriptor = &descriptors[*typed_descriptor_indices
                                    .get(&(instance, shard, producer, matrix.clone()))
                                    .ok_or("prepared variant source descriptor is unavailable")?];
                                let header = allocate_prepared_matrix(
                                    &region_storages,
                                    *binding_map
                                        .get(&source_descriptor.binding)
                                        .ok_or("prepared variant source binding is unavailable")?,
                                    &source_descriptor.params,
                                    matrix.rows,
                                    matrix.columns,
                                    source_descriptor.level,
                                    source_descriptor.is_ntt,
                                )?;
                                owners.insert(*wire, header);
                            }
                        }
                        if matches!(operation, PreparedGpuOperation::PreimageSample) {
                            let public = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("prepared preimage public owner missing")?;
                            let target = prepared_owner_for_wire(&owners, &aliases, arguments[2])
                                .ok_or("prepared preimage target owner missing")?;
                            let (secret, input) = secrets
                                .get(&arguments[1])
                                .ok_or("prepared preimage secret owner missing")?;
                            let mxx_ir_core::types::ConcreteWireType::Trapdoor { sigma, .. } =
                                &program.wire_types[&arguments[1]]
                            else {
                                return Err("prepared preimage secret is not trapdoor typed".into());
                            };
                            let sigma = sigma
                                .evaluate_f64(&node_source.environment)
                                .map_err(|error| error.to_string())?;
                            let compact_index =
                                compact_descriptor_indices[&(instance, shard, *node_id, variant)];
                            let compact = if let Some(owner) = allocated_compact.get(&compact_index)
                            {
                                Arc::clone(owner)
                            } else {
                                let layout = &compact_descriptors[compact_index];
                                let owner = allocate_prepared_compact(
                                    &region_storages,
                                    compact_bindings[compact_index],
                                    &layout.params,
                                    layout.rows,
                                    layout.columns,
                                    layout.bound.clone(),
                                )?;
                                allocated_compact.insert(compact_index, Arc::clone(&owner));
                                owner
                            };
                            let claims = GpuPreparedPreimageSampler::allocation_claims(
                                &descriptor.params,
                                public.row_size(),
                                &compact,
                            )?;
                            let command = bind_prepared_claims(
                                backend,
                                &mut region,
                                &descriptor.params,
                                source.device_id,
                                &claims,
                                || {
                                    GpuPreparedPreimageSampler::bind(
                                        &descriptor.params,
                                        secret,
                                        public,
                                        target,
                                        Arc::clone(&compact),
                                        sigma,
                                        source.global_column_start,
                                    )
                                },
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::Preimage {
                                command, rng: sampler_rngs.entry((instance, *node_id, variant)).or_insert_with(rand::rngs::StdRng::from_os_rng).clone(), secret: Arc::clone(secret), input: *input, output: Arc::clone(&compact), device: source.device_id, start: source.global_column_start,
                            }));
                            if program.outputs.contains(&output_wire) {
                                small_output_indices[instance].push(command_index);
                            }
                            compact_typed_owners.insert(
                                (output_wire, program.wire_types[&output_wire].clone()),
                                Arc::clone(&compact),
                            );
                            compact_owners.insert(output_wire, compact);
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::HashCompactDecompose) {
                            let scratch = allocate_prepared_matrix(
                                &region_storages,
                                *binding_map
                                    .get(&descriptor.binding)
                                    .ok_or("generic compact scratch binding missing")?,
                                &descriptor.params,
                                descriptor.rows,
                                descriptor.columns,
                                descriptor.level,
                                false,
                            )?;
                            let output_type = output_types
                                .and_then(|types| types.first())
                                .unwrap_or(&program.wire_types[&output_wire]);
                            let graph_node = &node_source;
                            let NodeKind::HashSample {
                                variant: hash_variant,
                                tag_prefix,
                                tag_components,
                                ..
                            } = graph_node.kind()
                            else {
                                return Err("generic compact hash node kind mismatch".into());
                            };
                            let small =
                                *hash_variant == mxx_ir_core::node::HashVariant::SmallDecomposed;
                            let digits = match graph_node.kind() {
                                NodeKind::HashSample { digit_count, .. } => digit_count
                                    .as_ref()
                                    .ok_or("generic compact digit count missing")?
                                    .evaluate(&node_source.environment)
                                    .map_err(|error| error.to_string())?
                                    .to_usize()
                                    .ok_or("generic compact digit count is not usize")?,
                                _ => return Err("generic compact hash node kind mismatch".into()),
                            };
                            let compact_index =
                                compact_descriptor_indices[&(instance, shard, *node_id, variant)];
                            let compact = if let Some(owner) = allocated_compact.get(&compact_index)
                            {
                                Arc::clone(owner)
                            } else {
                                let layout = &compact_descriptors[compact_index];
                                let owner = allocate_prepared_compact(
                                    &region_storages,
                                    compact_bindings[compact_index],
                                    &layout.params,
                                    layout.rows,
                                    layout.columns,
                                    layout.bound.clone(),
                                )?;
                                allocated_compact.insert(compact_index, Arc::clone(&owner));
                                owner
                            };
                            let arguments = &program
                                .node_bindings
                                .get(node_id)
                                .ok_or("generic compact hash arguments missing")?
                                .0;
                            let mut tag = tag_prefix.clone();
                            let mut operand_inputs = Vec::new();
                            for component in tag_components {
                                use mxx_ir_core::node::HashTagComponent;
                                match component {
                                    HashTagComponent::Bytes(bytes) => {
                                        tag.push(0);
                                        tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(bytes.as_slice());
                                    }
                                    HashTagComponent::Integer(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?;
                                        tag.push(1);
                                        append_hash_tag_integer(&mut tag, &value);
                                    }
                                    HashTagComponent::Operand(index) => {
                                        let wire = *arguments
                                            .get(*index)
                                            .ok_or("generic compact operand index out of range")?;
                                        operand_inputs.push(
                                            program
                                                .inputs
                                                .iter()
                                                .position(|candidate| *candidate == wire)
                                                .ok_or("generic compact operand is not an input")?,
                                        );
                                    }
                                    HashTagComponent::Decimal(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_string();
                                        tag.push(2);
                                        tag.extend_from_slice(&(value.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(value.as_bytes());
                                    }
                                    HashTagComponent::U64Le(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_u64()
                                            .ok_or("generic compact tag is not u64")?;
                                        tag.push(3);
                                        tag.extend_from_slice(&value.to_le_bytes());
                                    }
                                }
                            }
                            let key_wire =
                                *arguments.first().ok_or("generic compact key missing")?;
                            let key_input = program
                                .inputs
                                .iter()
                                .position(|wire| *wire == key_wire)
                                .ok_or("generic compact key is not an input")?;
                            let hash = GpuPreparedHashSample::bind(
                                Arc::clone(&scratch),
                                [0; 32],
                                &tag,
                                GpuMatrixSampleDist::Uniform,
                                0.0,
                                descriptor.params.modulus().to_u64().unwrap_or(0).saturating_sub(1),
                                descriptor.columns,
                                source.global_column_start,
                                None,
                            )?;
                            let decompose = GpuPreparedCompactDecompose::bind(
                                Arc::clone(&scratch),
                                Arc::clone(&compact),
                                small,
                                Some(digits),
                            )?;
                            let hash_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::hash_sample(
                                hash,
                                key_input,
                                operand_inputs.into_boxed_slice(),
                                tag.into_boxed_slice(),
                                Arc::clone(&scratch),
                                source.device_id,
                                source.global_column_start,
                            ));
                            let compact_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::compact_decompose(
                                decompose,
                                Arc::clone(&compact),
                                source.device_id,
                                source.global_column_start,
                            ));
                            if let Some(topology_node) =
                                program.topology.nodes.iter().find(|node| node.id == *node_id)
                            {
                                instances[instance][hash_index].apply_topology(topology_node);
                                instances[instance][compact_index].apply_topology(topology_node);
                                instances[instance][compact_index].wait_events =
                                    Box::new([topology_node.completion]);
                            }
                            if program.outputs.contains(&output_wire) {
                                small_output_indices[instance].push(compact_index);
                            }
                            compact_typed_owners
                                .insert((output_wire, output_type.clone()), Arc::clone(&compact));
                            compact_owners.insert(output_wire, compact);
                            return Ok(());
                        }
                        let output = allocate_prepared_matrix(
                            &region_storages,
                            *binding_map
                                .get(&descriptor.binding)
                                .ok_or("generic output binding missing")?,
                            &descriptor.params,
                            location.rows.end,
                            location.columns.end,
                            descriptor.level,
                            location.format ==
                                super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                        )?;
                        if matches!(operation, PreparedGpuOperation::TrapdoorSample) {
                            let NodeKind::TrapdoorSample { sigma, .. } = node_source.kind() else {
                                return Err("prepared trapdoor node mismatch".into());
                            };
                            let sigma = sigma
                                .evaluate_f64(&node_source.environment)
                                .map_err(|error| error.to_string())?;
                            let claims = GpuPreparedTrapdoorSampler::allocation_claims(
                                &descriptor.params,
                                output.row_size(),
                            );
                            let command = bind_prepared_claims(
                                backend,
                                &mut region,
                                &descriptor.params,
                                source.device_id,
                                &claims,
                                || {
                                    GpuPreparedTrapdoorSampler::bind(
                                        &descriptor.params,
                                        Arc::clone(&output),
                                        sigma,
                                    )
                                },
                            )?;
                            let secret_wire =
                                *outputs.get(1).ok_or("prepared trapdoor secret port missing")?;
                            secrets.insert(secret_wire, (Arc::clone(command.trapdoor()), None));
                            owners.insert(secret_wire, Arc::clone(&output));
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::Trapdoor { command, rng: sampler_rngs.entry((instance, *node_id, variant)).or_insert_with(rand::rngs::StdRng::from_os_rng).clone(), output: Arc::clone(&output), device: source.device_id, start: source.global_column_start }));
                            owners.insert(output_wire, output);
                            for wire in outputs {
                                if program.outputs.contains(wire) {
                                    output_indices[instance].push(command_index);
                                }
                            }
                            return Ok(());
                        }
                        if program.selection_commands.contains_key(node_id) {
                            let selection = program
                                .selection_commands
                                .get(node_id)
                                .ok_or("generic dynamic selection has no candidate table")?;
                            let candidates = match selection {
                                super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                                    candidates,
                                    ..
                                } |
                                super::gpu_prepared_lowering::PreparedSelection::Select {
                                    candidates,
                                    ..
                                } => candidates,
                                super::gpu_prepared_lowering::PreparedSelection::Static {
                                    ..
                                } => {
                                    return Err(
                                        "generic static selection was emitted as dynamic".into()
                                    );
                                }
                                super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { .. } |
                                super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { .. } |
                                super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { .. } => {
                                    return Err("scalar selection cannot be a matrix command".into());
                                }
                            };
                            let selector = match selection {
                                super::gpu_prepared_lowering::PreparedSelection::Dynamic {
                                    selector,
                                    ..
                                } |
                                super::gpu_prepared_lowering::PreparedSelection::Select {
                                    selector,
                                    ..
                                } => device_scalars.get(selector).cloned(),
                                _ => None,
                            };
                            let mut scalar_candidates = Vec::new();
                            let mut native_candidates = Vec::with_capacity(candidates.len());
                            for candidate in candidates.iter() {
                                let source = owners
                                    .iter()
                                    .find(|(wire, _)| {
                                        program.values.get(wire).is_some_and(|value| {
                                            value.owner == candidate.owner &&
                                                value.shape() == candidate.shape()
                                        })
                                    })
                                    .map(|(_, owner)| Arc::clone(owner))
                                    .ok_or("generic selection candidate owner is unavailable")?;
                                let view = GpuPreparedView {
                                    left: GpuPreparedRange {
                                        rows: candidate.rows.clone(),
                                        columns: candidate.columns.clone(),
                                    },
                                    right: GpuPreparedRange {
                                        rows: candidate.rows.clone(),
                                        columns: candidate.columns.clone(),
                                    },
                                    output: GpuPreparedRange {
                                        rows: location.rows.clone(),
                                        columns: location.columns.clone(),
                                    },
                                };
                                if selector.is_some() {
                                    scalar_candidates.push((source, view));
                                    continue;
                                }
                                let command = GpuPreparedInputCopy::bind(
                                    Arc::clone(&output),
                                    Arc::clone(&source),
                                    Some(view),
                                )?;
                                native_candidates
                                    .push(PreparedSelectionCandidate { command, source });
                            }
                            let command_index = instances[instance].len();
                            let shard_source = &source;
                            let mut command = if let Some(selector) = selector {
                                let claims =
                                    [GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                                        kind: GpuPreparedSlotKind::BatchWorkspace,
                                        bytes: GpuPreparedScalarMatrixSelect::workspace_bytes(
                                            scalar_candidates.len(),
                                        ),
                                        alignment: 8,
                                    })];
                                let plan = bind_prepared_claims(
                                    backend,
                                    &mut region,
                                    output.params(),
                                    shard_source.device_id,
                                    &claims,
                                    || {
                                        GpuPreparedScalarMatrixSelect::bind(
                                            Arc::clone(&output),
                                            selector,
                                            &scalar_candidates,
                                        )
                                    },
                                )?;
                                PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::ScalarMatrixSelect { command: plan, output: Arc::clone(&output), device: shard_source.device_id, start: shard_source.global_column_start })
                            } else {
                                PreparedCommand::selection(
                                    native_candidates.into_boxed_slice(),
                                    Arc::clone(&output),
                                    shard_source.device_id,
                                    shard_source.global_column_start,
                                )
                            };
                            if let Some(topology_node) =
                                program.topology.nodes.iter().find(|node| node.id == *node_id)
                            {
                                command.apply_topology(topology_node);
                            }
                            instances[instance].push(command);
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::PackPolynomialCoefficients) {
                            let NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } =
                                node_source.kind()
                            else {
                                unreachable!()
                            };
                            let bits = coefficient_bits
                                .evaluate(&node_source.environment)
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or("invalid packed width")?;
                            let mut scalar_wires = Vec::new();
                            for argument in arguments {
                                if matches!(
                                    program.wire_types.get(argument),
                                    Some(
                                        mxx_ir_core::types::ConcreteWireType::IndexedFamily { .. }
                                    )
                                ) {
                                    scalar_wires.extend(
                                        super::gpu_prepared_lowering::family_leaf_wires(
                                            program, *argument,
                                        ),
                                    );
                                } else {
                                    scalar_wires.push(*argument);
                                }
                            }
                            let values = scalar_wires
                                .iter()
                                .map(|wire| {
                                    device_scalars
                                        .get(wire)
                                        .cloned()
                                        .ok_or("packed scalar is not device-bound".to_owned())
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let claims = [
                                GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                                    kind: GpuPreparedSlotKind::BatchWorkspace,
                                    bytes: GpuPreparedScalarPack::workspace_bytes(values.len()),
                                    alignment: 8,
                                }),
                                GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                                    kind: GpuPreparedSlotKind::CompletionEvent,
                                    bytes: 0,
                                    alignment: 1,
                                }),
                            ];
                            let plan = bind_prepared_claims(
                                backend,
                                &mut region,
                                output.params(),
                                source.device_id,
                                &claims,
                                || GpuPreparedScalarPack::bind(Arc::clone(&output), &values, bits),
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::new(crate::backend::poly_gpu::gpu_prepared::PreparedOperation::ScalarPack { command: plan, output: Arc::clone(&output), device: source.device_id, start: source.global_column_start }));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(
                            operation,
                            PreparedGpuOperation::RnsUpload |
                                PreparedGpuOperation::LiftIntegerToConstantPolynomial
                        ) {
                            let input_wire =
                                *arguments.first().ok_or("generic RNS upload input is missing")?;
                            let input = program
                                .inputs
                                .iter()
                                .position(|wire| *wire == input_wire)
                                .ok_or("generic RNS upload input is not a root input")?;
                            let constant = matches!(
                                operation,
                                PreparedGpuOperation::LiftIntegerToConstantPolynomial
                            );
                            if constant {
                                if !matches!(
                                    runtime_inputs.get(input),
                                    Some(PreparedRuntimeValue::Int(_))
                                ) {
                                    return Err(
                                        "generic constant upload input is not an integer".into()
                                    );
                                }
                            } else if !matches!(
                                runtime_inputs.get(input),
                                Some(PreparedRuntimeValue::Bytes(_))
                            ) {
                                return Err("generic RNS upload input is not bytes".into());
                            }
                            let bytes_per_poly = (output.level() + 1)
                                .checked_mul(output.params().ring_dimension() as usize)
                                .and_then(|count| count.checked_mul(std::mem::size_of::<u64>()))
                                .ok_or("generic RNS upload byte stride overflow")?;
                            let format = match node_source.kind() {
                                NodeKind::PolynomialFromValues { evaluation: true, .. }
                                    if !constant =>
                                {
                                    GPU_POLY_FORMAT_EVAL
                                }
                                _ => GPU_POLY_FORMAT_COEFF,
                            };
                            let transform_to_eval =
                                output.is_ntt() && format == GPU_POLY_FORMAT_COEFF;
                            let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                                source: None,
                                target: Some(Arc::clone(&output)),
                                coefficient_index: 0,
                                coefficient_count: 0,
                                words_per_poly: 0,
                                bytes_per_poly,
                                format,
                                transform_to_eval,
                            };
                            let mut claims =
                                vec![GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                                    kind: GpuPreparedSlotKind::PinnedHost,
                                    bytes: bytes_per_poly * output.row_size() * output.col_size(),
                                    alignment: 1,
                                })];
                            for _ in 0..=output.level() {
                                claims.push(GpuTracedClaim::workspace(
                                    GpuPreparedWorkspaceLayout {
                                        kind: GpuPreparedSlotKind::TransferWorkspace,
                                        bytes: output.row_size() *
                                            output.col_size() *
                                            output.params().ring_dimension() as usize *
                                            8,
                                        alignment: 8,
                                    },
                                ));
                                claims.push(GpuTracedClaim::workspace(
                                    GpuPreparedWorkspaceLayout {
                                        kind: GpuPreparedSlotKind::CompletionEvent,
                                        bytes: 0,
                                        alignment: 1,
                                    },
                                ));
                            }
                            let host_command = bind_prepared_claims(
                                backend,
                                &mut region,
                                output.params(),
                                source.device_id,
                                &claims,
                                || super::gpu_prepared_host::bind_upload(&spec),
                            )?;
                            let super::gpu_prepared_host::PreparedHostCommand::Upload { command } =
                                host_command
                            else {
                                return Err(
                                    "prepared RNS upload binding returned wrong command".into()
                                );
                            };
                            let command_index = instances[instance].len();
                            let mut command = if constant {
                                PreparedCommand::upload_constant(
                                    command,
                                    input,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                )
                            } else {
                                PreparedCommand::upload(
                                    command,
                                    input,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                )
                            };
                            if let Some(topology_node) =
                                program.topology.nodes.iter().find(|node| node.id == *node_id)
                            {
                                command.apply_topology(topology_node);
                            }
                            instances[instance].push(command);
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::FixedCopies) {
                            let active_locations = arguments
                                .iter()
                                .map(|wire| {
                                    let mut location = program.values[wire].clone();
                                    let owner =
                                        prepared_owner_for_wire(&owners, &aliases, *wire)
                                            .ok_or("prepared view source owner is unavailable")?;
                                    location.rows = 0..owner.row_size();
                                    location.columns = 0..owner.col_size();
                                    Ok(location)
                                })
                                .collect::<Result<Vec<_>, String>>()?;
                            let resolved_view;
                            let view = if matches!(node_source.kind(), NodeKind::Slice { .. }) {
                                resolved_view = super::gpu_prepared_lowering::lower_slice(
                                    node_source.kind(),
                                    &node_source.environment,
                                    active_locations[0].clone(),
                                    location.clone(),
                                )
                                .map_err(|error| format!("prepared slice variant: {error:?}"))?;
                                &resolved_view
                            } else if matches!(node_source.kind(), NodeKind::Concat { .. }) {
                                resolved_view = super::gpu_prepared_lowering::lower_concat(
                                    node_source.kind(),
                                    &active_locations,
                                    location.clone(),
                                )
                                .map_err(|error| format!("prepared concat variant: {error:?}"))?;
                                &resolved_view
                            } else {
                                program
                                    .view_commands
                                    .get(node_id)
                                    .ok_or("generic fixed-copy node has no prepared view")?
                            };
                            let super::gpu_prepared_lowering::PreparedView::FixedCopies(copies) =
                                view
                            else {
                                return Err("generic fixed-copy node has an alias view".into());
                            };
                            let mut copy_commands = Vec::with_capacity(copies.len());
                            let mut copy_sources = Vec::with_capacity(copies.len());
                            for copy in copies.iter() {
                                let source_owner = prepared_owner_for_wire(
                                    &owners,
                                    &aliases,
                                    arguments
                                        .iter()
                                        .copied()
                                        .find(|wire| {
                                            program.values.get(wire).is_some_and(|location| {
                                                location.owner == copy.source.owner
                                            })
                                        })
                                        .ok_or("generic fixed-copy source owner missing")?,
                                )
                                .ok_or("generic fixed-copy source owner missing")?;
                                let command = GpuPreparedInputCopy::bind(
                                    Arc::clone(&output),
                                    Arc::clone(&source_owner),
                                    Some(GpuPreparedView {
                                        left: GpuPreparedRange {
                                            rows: copy.source.rows.clone(),
                                            columns: copy.source.columns.clone(),
                                        },
                                        right: GpuPreparedRange {
                                            rows: copy.source.rows.clone(),
                                            columns: copy.source.columns.clone(),
                                        },
                                        output: GpuPreparedRange {
                                            rows: copy.destination.rows.clone(),
                                            columns: copy.destination.columns.clone(),
                                        },
                                    }),
                                )?;
                                copy_commands.push(command);
                                copy_sources.push(source_owner);
                            }
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::concat_rows(
                                copy_commands.into_boxed_slice(),
                                copy_sources.into_boxed_slice(),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, Arc::clone(&output));
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(
                            operation,
                            PreparedGpuOperation::ModulusSwitch |
                                PreparedGpuOperation::ModulusReduce |
                                PreparedGpuOperation::CenteredExtend |
                                PreparedGpuOperation::BlockModSwitch |
                                PreparedGpuOperation::RnsModUp |
                                PreparedGpuOperation::RnsModDown
                        ) {
                            let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("generic conversion source owner missing")?;
                            let node = &node_source;
                            let plan = match node.kind() {
                                NodeKind::ModulusSwitch { .. } => {
                                    GpuPreparedModulusConversion::new(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::Round,
                                    )?
                                }
                                NodeKind::ModulusReduce { .. } => {
                                    GpuPreparedModulusConversion::new(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::Reduce,
                                    )?
                                }
                                NodeKind::CenteredExtend { .. } => {
                                    GpuPreparedModulusConversion::new(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::CenteredExtend,
                                    )?
                                }
                                NodeKind::BlockModSwitch { plaintext_modulus, .. } => {
                                    let plaintext_modulus = plaintext_modulus
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or("generic block switch modulus is not u64")?;
                                    GpuPreparedModulusConversion::new(
                                        &lhs,
                                        &output,
                                        GpuMatrixModulusConversion::BlockSwitch {
                                            plaintext_modulus,
                                        },
                                    )?
                                }
                                NodeKind::RnsModUp { digit_size, normalize, .. } => {
                                    GpuPreparedModulusConversion::new_rns_up(
                                        &lhs,
                                        &output,
                                        *digit_size,
                                        *normalize,
                                    )?
                                }
                                NodeKind::RnsModDown { plaintext_modulus, .. } => {
                                    let plaintext_modulus = plaintext_modulus
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or("generic RNS down modulus is not u64")?;
                                    GpuPreparedModulusConversion::new_rns_down(
                                        &lhs,
                                        &output,
                                        plaintext_modulus,
                                    )?
                                }
                                _ => {
                                    return Err("generic conversion operation kind mismatch".into())
                                }
                            };
                            let command =
                                GpuPreparedModulusConversion::bind(Arc::new(plan), lhs, &output)?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::modulus(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::MatrixMulSmallRhs) {
                            let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("generic compact multiplication lhs owner missing")?;
                            let rhs_wire = *arguments
                                .get(1)
                                .ok_or("generic compact multiplication rhs is missing")?;
                            let rhs_owner = compact_owners.get(&rhs_wire).cloned();
                            let rhs_fleet = compact_inputs.get(&rhs_wire);
                            let rhs = rhs_fleet
                                .and_then(|fleet| {
                                    fleet
                                        .shards()
                                        .iter()
                                        .find(|shard| shard.device_id == source.device_id)
                                })
                                .map(|shard| Arc::clone(&shard.value));
                            let rhs = rhs_owner.or(rhs);
                            let rhs =
                                rhs.ok_or("generic compact multiplication rhs is not compact")?;
                            let claims = lhs
                                .params()
                                .small_rhs_workspaces(lhs.level(), rhs.size().0, rhs.size().1)?
                                .into_iter()
                                .map(GpuTracedClaim::workspace)
                                .collect::<Vec<_>>();
                            let command = bind_prepared_claims(
                                backend,
                                &mut region,
                                lhs.params(),
                                source.device_id,
                                &claims,
                                || {
                                    GpuPreparedSmallRhs::bind(
                                        Arc::clone(&output),
                                        Arc::clone(&lhs),
                                        Arc::clone(&rhs),
                                        source.value.params().vram_budget_bytes(),
                                    )
                                },
                            )?;
                            let command_index = instances[instance].len();
                            if compact_owners.contains_key(&rhs_wire) {
                                instances[instance].push(PreparedCommand::small_rhs_from_owner(
                                    command,
                                    Arc::clone(&lhs),
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                ));
                            } else {
                                let lhs_index = matrix_input_wires
                                    .iter()
                                    .position(|wire| *wire == arguments[0])
                                    .ok_or(
                                        "generic compact multiplication lhs is not a root input",
                                    )?;
                                instances[instance].push(PreparedCommand::small_rhs(
                                    command,
                                    lhs_index * shard_count + shard,
                                    Arc::clone(&output),
                                    source.device_id,
                                    source.global_column_start,
                                ));
                            }
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::HashSample) {
                            let node = &node_source;
                            let NodeKind::HashSample { tag_prefix, tag_components, .. } =
                                node.kind()
                            else {
                                return Err("generic hash operation kind mismatch".into());
                            };
                            if !matches!(
                                node.kind(),
                                NodeKind::HashSample {
                                    variant: mxx_ir_core::node::HashVariant::Plain,
                                    base: None,
                                    digit_count: None,
                                    ..
                                }
                            ) {
                                return Err(
                            "prepared hash command requires the plain non-decomposed variant"
                                .into(),
                        );
                            }
                            let mut tag = tag_prefix.clone();
                            let mut operand_inputs = Vec::new();
                            for component in tag_components {
                                use mxx_ir_core::node::HashTagComponent;
                                match component {
                                    HashTagComponent::Bytes(bytes) => {
                                        tag.push(0);
                                        tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(bytes);
                                    }
                                    HashTagComponent::Integer(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?;
                                        tag.push(1);
                                        append_hash_tag_integer(&mut tag, &value);
                                    }
                                    HashTagComponent::Decimal(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_string();
                                        tag.push(2);
                                        tag.extend_from_slice(&(value.len() as u64).to_be_bytes());
                                        tag.extend_from_slice(value.as_bytes());
                                    }
                                    HashTagComponent::U64Le(expression) => {
                                        let value = expression
                                            .evaluate(&node_source.environment)
                                            .map_err(|error| error.to_string())?
                                            .to_u64()
                                            .ok_or("prepared hash tag integer is not u64")?;
                                        tag.push(3);
                                        tag.extend_from_slice(&value.to_le_bytes());
                                    }
                                    HashTagComponent::Operand(_) => {
                                        let HashTagComponent::Operand(argument_index) = component
                                        else {
                                            unreachable!();
                                        };
                                        let wire = *arguments
                                            .get(*argument_index)
                                            .ok_or("prepared hash operand index is out of range")?;
                                        let input = program
                                            .inputs
                                            .iter()
                                            .position(|candidate| *candidate == wire)
                                            .ok_or("prepared hash operand is not a root input")?;
                                        operand_inputs.push(input);
                                    }
                                }
                            }
                            let key_wire =
                                *arguments.first().ok_or("generic hash key is missing")?;
                            let input = program
                                .inputs
                                .iter()
                                .position(|wire| *wire == key_wire)
                                .ok_or("generic hash key is not a root input")?;
                            let command = GpuPreparedHashSample::bind(
                                Arc::clone(&output),
                                [0; 32],
                                &tag,
                                GpuMatrixSampleDist::Uniform,
                                0.0,
                                output.params().modulus().to_u64().unwrap_or(0).saturating_sub(1),
                                source.value.col_size(),
                                source.global_column_start,
                                None,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::hash_sample(
                                command,
                                input,
                                operand_inputs.clone().into_boxed_slice(),
                                tag.into_boxed_slice(),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(
                            operation,
                            PreparedGpuOperation::UniformResidueSample |
                                PreparedGpuOperation::UniformIntervalSample |
                                PreparedGpuOperation::GaussianSample
                        ) {
                            let node = &node_source;
                            let (dist, sigma, bound) = match node.kind() {
                                NodeKind::UniformResidueSample { .. } => (
                                    GpuMatrixSampleDist::Uniform,
                                    0.0,
                                    output
                                        .params()
                                        .modulus()
                                        .to_u64()
                                        .unwrap_or(0)
                                        .saturating_sub(1),
                                ),
                                NodeKind::UniformIntervalSample { range, .. } => {
                                    let minimum = range
                                        .minimum
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_i64()
                                        .ok_or("generic uniform interval minimum is not i64")?;
                                    let maximum = range
                                        .maximum
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_i64()
                                        .ok_or("generic uniform interval maximum is not i64")?;
                                    (
                                        GpuMatrixSampleDist::Uniform,
                                        0.0,
                                        maximum.unsigned_abs().max(minimum.unsigned_abs()),
                                    )
                                }
                                NodeKind::GaussianSample {
                                    sigma, max_coefficient_bound, ..
                                } => (
                                    GpuMatrixSampleDist::Gauss,
                                    sigma
                                        .evaluate_f64(&node_source.environment)
                                        .map_err(|error| error.to_string())?,
                                    max_coefficient_bound
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or("generic gaussian bound is not u64")?,
                                ),
                                _ => return Err("generic sampling operation kind mismatch".into()),
                            };
                            let command = GpuPreparedSampling::bind(
                                Arc::clone(&output),
                                dist,
                                sigma,
                                bound,
                                source.value.col_size(),
                                source.global_column_start,
                                None,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::sampling(
                                command,
                                mxx_primitives::poly::dcrt::gpu::GpuRngSeed::from_bytes([0; 32]),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::MatrixMulAccumulate) {
                            let node = &node_source;
                            let NodeKind::MatrixMulAccumulate { coefficients, has_bias } =
                                node.kind()
                            else {
                                return Err("generic accumulate operation kind mismatch".into());
                            };
                            let mut terms = Vec::with_capacity(coefficients.len());
                            for (index, coefficient) in coefficients.iter().enumerate() {
                                let left = prepared_owner_for_wire(
                                    &owners,
                                    &aliases,
                                    arguments[2 * index],
                                )
                                .ok_or("generic accumulate lhs owner missing")?;
                                let right = prepared_owner_for_wire(
                                    &owners,
                                    &aliases,
                                    arguments[2 * index + 1],
                                )
                                .ok_or("generic accumulate rhs owner missing")?;
                                let scalar = coefficient
                                    .evaluate(&node_source.environment)
                                    .map_err(|error| error.to_string())?
                                    .to_u64()
                                    .ok_or("generic accumulate coefficient is not u64")?;
                                let residues = source
                                    .value
                                    .params()
                                    .moduli()
                                    .iter()
                                    .map(|prime| scalar % prime)
                                    .collect();
                                terms.push((left, right, residues));
                            }
                            let bias = if *has_bias {
                                Some(
                                    prepared_owner_for_wire(
                                        &owners,
                                        &aliases,
                                        *arguments
                                            .last()
                                            .ok_or("generic accumulate bias missing")?,
                                    )
                                    .ok_or("generic accumulate bias owner missing")?,
                                )
                            } else {
                                None
                            };
                            let command = GpuPreparedAccumulateCommand::bind(
                                terms,
                                bias,
                                Arc::clone(&output),
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::accumulate(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::ConcatRows) {
                            if arguments.len() != 2 {
                                return Err("generic row concat requires two inputs".into());
                            }
                            let sources = arguments
                                .iter()
                                .map(|wire| {
                                    prepared_owner_for_wire(&owners, &aliases, *wire)
                                        .ok_or("generic concat owner missing")
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let mut row_offset = 0;
                            let mut commands = Vec::with_capacity(sources.len());
                            for source_owner in &sources {
                                let rows = source_owner.row_size();
                                let columns = source_owner.col_size();
                                let view = GpuPreparedView {
                                    left: GpuPreparedRange { rows: 0..rows, columns: 0..columns },
                                    right: GpuPreparedRange { rows: 0..rows, columns: 0..columns },
                                    output: GpuPreparedRange {
                                        rows: row_offset..row_offset + rows,
                                        columns: 0..columns,
                                    },
                                };
                                commands.push(GpuPreparedInputCopy::bind(
                                    Arc::clone(&output),
                                    Arc::clone(source_owner),
                                    Some(view),
                                )?);
                                row_offset += rows;
                            }
                            if row_offset != output.row_size() {
                                return Err("generic row concat output shape mismatch".into());
                            }
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::concat_rows(
                                commands.into_boxed_slice(),
                                sources.into_boxed_slice(),
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::Transpose) {
                            let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                                .ok_or("generic transpose source owner missing")?;
                            if lhs.row_size() != output.col_size() ||
                                lhs.col_size() != output.row_size()
                            {
                                return Err("generic transpose output shape mismatch".into());
                            }
                            let command =
                                mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedTranspose::bind(
                                    lhs,
                                    Arc::clone(&output),
                                    None,
                                )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::transpose(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        let lhs = prepared_owner_for_wire(&owners, &aliases, arguments[0])
                            .ok_or("generic lhs owner missing")?;
                        let rhs = arguments
                            .get(1)
                            .and_then(|wire| prepared_owner_for_wire(&owners, &aliases, *wire));
                        if matches!(operation, PreparedGpuOperation::CenteredRebase) {
                            let command = GpuPreparedCenteredRebase::bind(
                                Arc::clone(&lhs),
                                Arc::clone(&output),
                                None,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::centered_rebase(
                                command,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::GadgetDecompose) {
                            let NodeKind::GadgetDecompose { digit_count, small, .. } =
                                node_source.kind()
                            else {
                                return Err("generic gadget operation kind mismatch".into());
                            };
                            let digits = digit_count
                                .evaluate(&node_source.environment)
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or("generic gadget digit count is not usize")?;
                            let compact_index = compact_descriptor_indices
                                .get(&(instance, shard, *node_id, 0))
                                .copied()
                                .ok_or("generic gadget compact descriptor is missing")?;
                            let layout = &compact_descriptors[compact_index];
                            let compact = allocate_prepared_compact(
                                &region_storages,
                                compact_bindings[compact_index],
                                &layout.params,
                                layout.rows,
                                layout.columns,
                                layout.bound.clone(),
                            )?;
                            let command = GpuPreparedCompactDecompose::bind(
                                Arc::clone(&lhs),
                                Arc::clone(&compact),
                                *small,
                                Some(digits),
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::compact_decompose(
                                command,
                                Arc::clone(&compact),
                                source.device_id,
                                source.global_column_start,
                            ));
                            compact_owners.insert(output_wire, compact);
                            if program.outputs.contains(&output_wire) {
                                small_output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        if matches!(operation, PreparedGpuOperation::CrtRecompose) {
                            let node = &node_source;
                            let NodeKind::CrtRecompose {
                                plaintext_moduli,
                                reconstruction_coefficients,
                                ..
                            } = node.kind()
                            else {
                                return Err("generic CRT operation kind mismatch".into());
                            };
                            if arguments.len() != plaintext_moduli.len() ||
                                arguments.len() != reconstruction_coefficients.len()
                            {
                                return Err("generic CRT level count mismatch".into());
                            }
                            let levels = arguments
                                .iter()
                                .map(|wire| {
                                    prepared_owner_for_wire(&owners, &aliases, *wire)
                                        .ok_or("generic CRT level owner missing")
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let levels: Arc<[Arc<GpuDCRTPolyMatrix>]> = Arc::from(levels);
                            let plaintext_moduli = plaintext_moduli
                                .iter()
                                .map(|value| {
                                    value
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or_else(|| {
                                            "generic CRT plaintext modulus is not u64".to_owned()
                                        })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let reconstruction_coefficients = reconstruction_coefficients
                                .iter()
                                .map(|value| {
                                    value
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_u64()
                                        .ok_or_else(|| {
                                            "generic CRT reconstruction coefficient is not u64"
                                                .to_owned()
                                        })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let command = GpuPreparedCrtRecompose::bind(
                                Arc::clone(&output),
                                Arc::clone(&levels),
                                plaintext_moduli,
                                reconstruction_coefficients,
                            )?;
                            let command_index = instances[instance].len();
                            instances[instance].push(PreparedCommand::crt_recompose(
                                command,
                                levels,
                                Arc::clone(&output),
                                source.device_id,
                                source.global_column_start,
                            ));
                            owners.insert(output_wire, output);
                            if program.outputs.contains(&output_wire) {
                                output_indices[instance].push(command_index);
                            }
                            return Ok(());
                        }
                        let kind = match operation {
                            PreparedGpuOperation::MatrixBinary(MatrixBinaryOp::Add) => {
                                GpuPreparedArithmeticKind::Add
                            }
                            PreparedGpuOperation::MatrixBinary(MatrixBinaryOp::Subtract) => {
                                GpuPreparedArithmeticKind::Subtract
                            }
                            PreparedGpuOperation::MatrixBinary(MatrixBinaryOp::Multiply) => {
                                GpuPreparedArithmeticKind::Multiply
                            }
                            PreparedGpuOperation::Tensor => GpuPreparedArithmeticKind::Tensor,
                            PreparedGpuOperation::MatrixNegate => GpuPreparedArithmeticKind::Negate,
                            PreparedGpuOperation::MatrixScale => {
                                let node = &node_source;
                                let NodeKind::MatrixScale { scalar } = node.kind() else {
                                    return Err("generic scale operation kind mismatch".into());
                                };
                                let value = scalar
                                    .evaluate(&node_source.environment)
                                    .map_err(|error| error.to_string())?
                                    .to_u64()
                                    .ok_or("generic scale is not u64")?;
                                GpuPreparedArithmeticKind::Scale {
                                    residues: source
                                        .value
                                        .params()
                                        .moduli()
                                        .iter()
                                        .map(|prime| value % prime)
                                        .collect(),
                                }
                            }
                            PreparedGpuOperation::RingAutomorphism => {
                                let node = &node_source;
                                let NodeKind::RingAutomorphism { index } = node.kind() else {
                                    return Err(
                                        "generic automorphism operation kind mismatch".into()
                                    );
                                };
                                GpuPreparedArithmeticKind::Automorphism {
                                    index: index
                                        .evaluate(&node_source.environment)
                                        .map_err(|error| error.to_string())?
                                        .to_usize()
                                        .ok_or("generic automorphism is not usize")?,
                                }
                            }
                            _ => {
                                return Err(
                                    "generic operation is not a matrix arithmetic kind".into()
                                )
                            }
                        };
                        let command =
                            GpuPreparedArithmetic::bind(kind, lhs, rhs, Arc::clone(&output))?;
                        let command_index = instances[instance].len();
                        let topology_node = program
                            .topology
                            .nodes
                            .iter()
                            .find(|node| node.id == *node_id)
                            .ok_or("generic topology node disappeared")?;
                        let mut prepared_command = PreparedCommand::arithmetic(
                            command,
                            Arc::clone(&output),
                            source.device_id,
                            source.global_column_start,
                        );
                        prepared_command.stream = topology_node.stream;
                        prepared_command.wait_events = topology_node.waits.clone();
                        prepared_command.completion_event = topology_node.completion;
                        instances[instance].push(prepared_command);
                        owners.insert(output_wire, output);
                        if program.outputs.contains(&output_wire) {
                            output_indices[instance].push(command_index);
                        }
                        Ok(())
                    })();
                    let topology = program
                        .topology
                        .nodes
                        .iter()
                        .find(|node| node.id == *node_id)
                        .ok_or("prepared native node has no topology identity")?;
                    for command in &mut instances[instance][first_command..] {
                        command.apply_topology(topology);
                        command.variant = variant;
                    }
                    result.map_err(|error| {
                        format!("prepared {operation:?} node {node_id}: {error}")
                    })?;
                    if variant != 0 {
                        output_indices[instance].truncate(previous_outputs);
                        small_output_indices[instance].truncate(previous_small_outputs);
                    }
                }
            }
            for (node_id, host_operation) in &host_nodes {
                if matches!(host_operation, PreparedGpuOperation::ThresholdDecode) {
                    continue;
                }
                let arguments = &program
                    .node_bindings
                    .get(node_id)
                    .ok_or("generic host node binding missing")?
                    .0;
                let source_wire = *arguments.first().ok_or("generic host source missing")?;
                let source_owner = prepared_owner_for_wire(&owners, &aliases, source_wire)
                    .ok_or("generic host source owner missing")?;
                let source_owner = if source_owner.is_ntt() {
                    let descriptor_index = host_staging_descriptor_indices
                        .get(&(instance, shard, *node_id))
                        .copied()
                        .ok_or("generic host coefficient staging descriptor is missing")?;
                    let descriptor = &descriptors[descriptor_index];
                    let binding = *binding_map
                        .get(&descriptor.binding)
                        .ok_or("generic host coefficient staging binding is missing")?;
                    let staging = allocate_prepared_matrix(
                        &region_storages,
                        binding,
                        &descriptor.params,
                        descriptor.rows,
                        descriptor.columns,
                        descriptor.level,
                        false,
                    )?;
                    let first_command = instances[instance].len();
                    let copy = GpuPreparedInputCopy::bind(
                        Arc::clone(&staging),
                        Arc::clone(&source_owner),
                        None,
                    )?;
                    instances[instance].push(PreparedCommand::input_copy_from_owner(
                        copy,
                        Arc::clone(&source_owner),
                        Arc::clone(&staging),
                        source.device_id,
                        source.global_column_start,
                    ));
                    let inverse = GpuPreparedTransform::new_inverse(&staging)?;
                    instances[instance].push(PreparedCommand::transform(
                        inverse,
                        Arc::clone(&staging),
                        source.device_id,
                        source.global_column_start,
                    ));
                    if let Some(topology_node) =
                        program.topology.nodes.iter().find(|node| node.id == *node_id)
                    {
                        for command in &mut instances[instance][first_command..] {
                            command.apply_topology(topology_node);
                        }
                    }
                    staging
                } else {
                    source_owner
                };
                let (coefficient_index, coefficient_count) = match host_operation {
                    PreparedGpuOperation::ExtractCoefficient => {
                        let NodeKind::ExtractCoefficient { position, .. } =
                            program.node_sources[node_id].kind()
                        else {
                            return Err("prepared coefficient readback node kind mismatch".into());
                        };
                        (
                            position
                                .evaluate(&program.node_sources[node_id].environment)
                                .map_err(|error| error.to_string())?
                                .to_usize()
                                .ok_or("prepared coefficient position is not usize")?,
                            1,
                        )
                    }
                    PreparedGpuOperation::RnsReadback => {
                        (0, source_owner.params().ring_dimension() as usize)
                    }
                    _ => return Err("prepared host operation kind mismatch".into()),
                };
                if matches!(host_operation, PreparedGpuOperation::ExtractCoefficient) {
                    let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                        source: Some(Arc::clone(&source_owner)),
                        target: None,
                        coefficient_index,
                        coefficient_count,
                        words_per_poly: source_owner.level() + 1,
                        bytes_per_poly: 0,
                        format: GPU_POLY_FORMAT_COEFF,
                        transform_to_eval: false,
                    };
                    let claims =
                        prepared_coeff_readback_claims(&source_owner, spec.words_per_poly)?;
                    let host_command = bind_prepared_claims(
                        backend,
                        &mut region,
                        source_owner.params(),
                        source.device_id,
                        &claims,
                        || super::gpu_prepared_host::bind_readback(&spec),
                    )?;
                    let super::gpu_prepared_host::PreparedHostCommand::Readback {
                        command: plan,
                        values,
                    } = host_command
                    else {
                        return Err(
                            "prepared coefficient readback binding returned wrong command".into()
                        );
                    };
                    let mut command = PreparedCommand::readback(
                        plan,
                        values,
                        *node_id,
                        source.device_id,
                        source.global_column_start,
                    );
                    if let Some(topology_node) =
                        program.topology.nodes.iter().find(|node| node.id == *node_id)
                    {
                        command.apply_topology(topology_node);
                    }
                    instances[instance].push(command);
                    continue;
                }
                let spec = super::gpu_prepared_host::PreparedHostCommandSpec {
                    source: Some(Arc::clone(&source_owner)),
                    target: None,
                    coefficient_index: 0,
                    coefficient_count,
                    words_per_poly: (source_owner.level() + 1)
                        .checked_mul(coefficient_count)
                        .ok_or("prepared reconstruction readback size overflow")?,
                    bytes_per_poly: 0,
                    format: GPU_POLY_FORMAT_COEFF,
                    transform_to_eval: false,
                };
                let claims = prepared_coeff_readback_claims(&source_owner, spec.words_per_poly)?;
                let host_command = bind_prepared_claims(
                    backend,
                    &mut region,
                    source_owner.params(),
                    source.device_id,
                    &claims,
                    || super::gpu_prepared_host::bind_reconstruction(&spec),
                )?;
                let super::gpu_prepared_host::PreparedHostCommand::Reconstruction {
                    command: plan,
                    ..
                } = host_command
                else {
                    return Err("prepared host reconstruction binding returned wrong command".into());
                };
                let values = Arc::new(Mutex::new(
                    plan.with_values(|values| values.to_vec().into_boxed_slice()),
                ));
                let mut command = PreparedCommand::reconstruction(
                    Arc::clone(&plan),
                    values,
                    *node_id,
                    source.device_id,
                    source.global_column_start,
                );
                if let Some(topology_node) =
                    program.topology.nodes.iter().find(|node| node.id == *node_id)
                {
                    command.stream = topology_node.stream;
                    command.wait_events = topology_node.waits.clone();
                    command.completion_event = topology_node.completion;
                }
                instances[instance].push(command);
            }
            for output_wire in &program.outputs {
                if !aliases.contains_key(output_wire) {
                    continue;
                }
                if let Some(output) = prepared_owner_for_wire(&owners, &aliases, *output_wire) {
                    let command_index = instances[instance].len();
                    instances[instance].push(PreparedCommand::alias(
                        output,
                        source.device_id,
                        source.global_column_start,
                    ));
                    if let Some(topology_node) = program.topology.nodes.iter().find(|node| {
                        node.id == output_wire.node.0 as u32 &&
                            matches!(
                                node.command.operation,
                                super::gpu_prepared_lowering::PreparedOperation::Alias
                            )
                    }) {
                        instances[instance]
                            .last_mut()
                            .expect("prepared alias command")
                            .apply_topology(topology_node);
                    }
                    output_indices[instance].push(command_index);
                }
            }
            // A sampler has public and secret ports, and exported ports need
            // not be in producer order. Freeze the exact export order now.
            let shard_outputs = &output_indices[instance][shard_output_begin..];
            let ordered = program
                .outputs
                .iter()
                .filter(|wire| {
                    matches!(
                        program.wire_types.get(wire),
                        Some(
                            mxx_ir_core::types::ConcreteWireType::Matrix(_) |
                                mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }
                        )
                    )
                })
                .map(|wire| {
                    let owner = prepared_owner_for_wire(&owners, &aliases, *wire)
                        .ok_or("prepared exported owner missing")?;
                    shard_outputs
                        .iter()
                        .copied()
                        .find(|index| Arc::ptr_eq(&instances[instance][*index].output().0, &owner))
                        .ok_or_else(|| "prepared exported command missing".to_owned())
                })
                .collect::<Result<Vec<_>, String>>()?;
            output_indices[instance].truncate(shard_output_begin);
            output_indices[instance].extend(ordered);
        }
    }

    provision_prepared_command_schedules(
        backend,
        &mut instances,
        &mut region,
        &program.topology.nodes,
    )?;
    let instances = instances
        .into_iter()
        .map(|commands| (commands.into_boxed_slice(), Vec::<usize>::new().into_boxed_slice()))
        .collect::<Vec<_>>();
    let execution = PreparedGpuFleetExecution::from_command_instances(
        instances,
        region,
        output_shape.0,
        output_shape.1,
    )
    .with_small_output_indices(
        small_output_indices.into_iter().map(|indices| indices.into_boxed_slice()).collect(),
    );
    for (instance, indices) in output_indices.into_iter().enumerate() {
        let fleet = &execution.pool.instances[instance];
        fleet.state.lock().expect("prepared GPU instance poisoned").output_commands =
            indices.into_boxed_slice();
    }
    Ok(execution.with_program(Arc::new(program.clone()))?)
}

#[derive(Clone)]
struct PreparedMatrixDescriptor {
    binding: super::gpu_prepared_lowering::PreparedBindingId,
    params: GpuDCRTPolyParams,
    device: i32,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
}

struct PreparedCompactDescriptor {
    params: GpuDCRTPolyParams,
    device: i32,
    rows: usize,
    columns: usize,
    bound: num_bigint::BigUint,
}

#[derive(Clone, Copy)]
struct PreparedCompactBinding {
    resources: [(u64, GpuPreparedRequest); 3],
}

fn select_prepared_compact(
    backend: &GpuDcrtBackend,
    inventory: &BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>,
    descriptor: &PreparedCompactDescriptor,
    used: &mut BTreeSet<(u64, u64, usize)>,
) -> Result<PreparedCompactBinding, String> {
    let bytes = GpuSmallMatrix::allocation_bytes(
        &descriptor.params,
        descriptor.rows,
        descriptor.columns,
        &descriptor.bound,
    )
    .map_err(|error| error.to_string())?;
    let mut resources = Vec::with_capacity(3);
    for (kind, bytes, alignment) in [
        (GpuPreparedSlotKind::CompactPayload, bytes, 256),
        (GpuPreparedSlotKind::PinnedHost, bytes, 1),
        (GpuPreparedSlotKind::CompletionEvent, 0, 1),
    ] {
        let mut selected = None;
        for (identity, (owner, storage)) in inventory {
            if storage.device() != descriptor.device ||
                storage.context_identity() != descriptor.params.context_identity() ||
                backend.prepared_device_index(descriptor.device) != Some(*owner)
            {
                continue;
            }
            let snapshots = storage.snapshot()?;
            for slot_index in 0..storage.slot_count() {
                let slot = storage.slot_identity(slot_index).ok_or("compact slot disappeared")?;
                if snapshots[slot_index].is_available() &&
                    slot.kind() == kind &&
                    slot.requested_backing_bytes() >= bytes
                {
                    let request = slot.workspace_request(bytes, alignment);
                    if used.insert(request.slot_key()) {
                        selected = Some((*identity, request));
                        break;
                    }
                }
            }
            if selected.is_some() {
                break;
            }
        }
        resources.push(selected.ok_or("accepted inventory has no compact owner/readback slot")?);
    }
    Ok(PreparedCompactBinding {
        resources: resources.try_into().map_err(|_| "compact resource count")?,
    })
}

fn allocate_prepared_compact(
    storages: &BTreeMap<u64, Arc<GpuPreparedStorage>>,
    binding: PreparedCompactBinding,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    bound: num_bigint::BigUint,
) -> Result<Arc<GpuSmallMatrix>, String> {
    let mut reservations = binding
        .resources
        .into_iter()
        .map(|(storage, request)| {
            storages
                .get(&storage)
                .ok_or("prepared compact storage missing".to_owned())?
                .reserve(std::slice::from_ref(&request))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let first = reservations.remove(0);
    let dispatch = first.enter(reservations)?;
    let mut output = GpuSmallMatrix::new_empty(params, rows, columns, bound)
        .map_err(|error| error.to_string())?;
    output.prepare_readback()?;
    drop(dispatch.finish()?);
    Ok(Arc::new(output))
}

fn prepared_binding_id(
    owner: u64,
    device: i32,
    instance: usize,
) -> super::gpu_prepared_lowering::PreparedBindingId {
    super::gpu_prepared_lowering::PreparedBindingId { owner, device, instance, storage: None }
}

#[derive(Clone, Copy)]
struct PreparedMatrixBinding {
    storage: u64,
    request: GpuPreparedRequest,
    identity: super::gpu_prepared_lowering::PreparedStorageBinding,
}

fn reserve_prepared_matrices(
    backend: &mut GpuDcrtBackend,
    descriptors: &[PreparedMatrixDescriptor],
) -> Result<
    (
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<u64, Arc<GpuPreparedStorage>>,
        BTreeMap<super::gpu_prepared_lowering::PreparedBindingId, PreparedMatrixBinding>,
    ),
    String,
> {
    let (region, storages, bindings, compact) =
        reserve_prepared_resources(backend, descriptors, &[])?;
    debug_assert!(compact.is_empty());
    Ok((region, storages, bindings))
}

fn bind_prepared_claims<T>(
    backend: &mut GpuDcrtBackend,
    region: &mut Arc<crate::gpu_memory::GpuMemoryRegion>,
    params: &GpuDCRTPolyParams,
    device: i32,
    claims: &[GpuTracedClaim],
    bind: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    backend
        .prepare_storage_claims(vec![(params.clone(), claims.to_vec())], true)
        .map_err(|error| error.to_string())?;
    let inventory = backend.prepared_storage_inventory().ok_or("prepared inventory missing")?;
    let mut selected = Vec::new();
    let mut used = BTreeSet::new();
    let mut requests = BTreeMap::<u64, Vec<GpuPreparedRequest>>::new();
    for claim in claims {
        let mut selection = None;
        for (id, (ledger_device, storage)) in &inventory {
            if storage.device() != device ||
                storage.context_identity() != params.context_identity() ||
                backend.prepared_device_index(device) != Some(*ledger_device)
            {
                continue;
            }
            for slot in storage.snapshot()? {
                if !slot.is_available() {
                    continue;
                }
                if let Some(request) = slot.request(params, claim)? {
                    if used.insert(request.slot_key()) {
                        selection = Some((*id, request));
                        break;
                    }
                }
            }
            if selection.is_some() {
                break;
            }
        }
        let (id, request) = selection.ok_or("accepted command resource missing")?;
        requests.entry(id).or_default().push(request);
        selected.push((id, request));
    }
    let (child, storages) =
        backend.reserve_standalone_prepared_region(&requests).map_err(|error| error.to_string())?;
    Arc::get_mut(region)
        .ok_or("prepared region published before binding")?
        .retain_detached_region(&child);
    // Preserve native consumption order, including repeated kinds across stores.
    let mut reservations = selected
        .into_iter()
        .map(|(id, request)| storages[&id].1.reserve(&[request]))
        .collect::<Result<Vec<_>, _>>()?;
    let first = reservations.remove(0);
    let dispatch = first.enter(reservations)?;
    let result = bind()?;
    drop(dispatch.finish()?);
    Ok(result)
}

fn prepared_coeff_readback_claims(
    source: &GpuDCRTPolyMatrix,
    words_per_poly: usize,
) -> Result<Vec<GpuTracedClaim>, String> {
    let (rows, columns) = source.size();
    let words = rows
        .checked_mul(columns)
        .and_then(|count| count.checked_mul(words_per_poly))
        .ok_or("prepared coefficient readback size overflow")?;
    let bytes = words
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or("prepared coefficient readback byte size overflow")?;
    let mut claims = vec![GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
        kind: GpuPreparedSlotKind::PinnedHost,
        bytes,
        alignment: std::mem::align_of::<u64>(),
    })];
    claims.extend((0..=source.level()).map(|_| {
        GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::CompletionEvent,
            bytes: 0,
            alignment: 1,
        })
    }));
    Ok(claims)
}

fn reserve_prepared_resources(
    backend: &mut GpuDcrtBackend,
    descriptors: &[PreparedMatrixDescriptor],
    compact_descriptors: &[PreparedCompactDescriptor],
) -> Result<
    (
        Arc<crate::gpu_memory::GpuMemoryRegion>,
        BTreeMap<u64, Arc<GpuPreparedStorage>>,
        BTreeMap<super::gpu_prepared_lowering::PreparedBindingId, PreparedMatrixBinding>,
        Vec<PreparedCompactBinding>,
    ),
    String,
> {
    backend.fence_released_memory().map_err(|error| error.to_string())?;
    backend.poll_prepared_releases().map_err(|error| error.to_string())?;
    let mut claims = BTreeMap::<usize, (GpuDCRTPolyParams, Vec<GpuTracedClaim>)>::new();
    for descriptor in descriptors {
        claims
            .entry(descriptor.params.context_identity())
            .or_insert_with(|| (descriptor.params.clone(), Vec::new()))
            .1
            .push(GpuTracedClaim::matrix(
                descriptor.rows,
                descriptor.columns,
                descriptor.level,
                descriptor.is_ntt,
            ));
    }
    for descriptor in compact_descriptors {
        let bytes = GpuSmallMatrix::allocation_bytes(
            &descriptor.params,
            descriptor.rows,
            descriptor.columns,
            &descriptor.bound,
        )
        .map_err(|error| error.to_string())?;
        claims
            .entry(descriptor.params.context_identity())
            .or_insert_with(|| (descriptor.params.clone(), Vec::new()))
            .1
            .extend([
                GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompactPayload,
                    bytes,
                    alignment: 256,
                }),
                GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::PinnedHost,
                    bytes,
                    alignment: 1,
                }),
                GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                }),
            ]);
    }
    backend
        .prepare_storage_claims(claims.into_values().collect(), true)
        .map_err(|error| error.to_string())?;
    let inventory = backend
        .prepared_storage_inventory()
        .ok_or("prepared graph has no accepted storage inventory")?;
    let mut used = BTreeSet::new();
    let mut requests = BTreeMap::<u64, Vec<GpuPreparedRequest>>::new();
    let mut bindings = BTreeMap::new();
    for descriptor in descriptors {
        let select = |exact: bool, used: &mut BTreeSet<(u64, u64, usize)>| {
            inventory.iter().find_map(|(identity, (owner, storage))| {
                if storage.device() != descriptor.device ||
                    storage.context_identity() != descriptor.params.context_identity() ||
                    backend.prepared_device_index(descriptor.device) != Some(*owner)
                {
                    return None;
                }
                let snapshots = storage.snapshot().ok()?;
                (0..storage.slot_count()).find_map(|slot_index| {
                    let slot = storage.slot_identity(slot_index)?;
                    if !snapshots[slot_index].is_available() ||
                        slot.kind() != GpuPreparedSlotKind::Matrix ||
                        slot.rows() < descriptor.rows ||
                        slot.columns() < descriptor.columns ||
                        slot.level() != Some(descriptor.level) ||
                        (exact &&
                            (slot.rows() != descriptor.rows ||
                                slot.columns() != descriptor.columns))
                    {
                        return None;
                    }
                    let request =
                        slot.matrix_request(descriptor.rows, descriptor.columns, descriptor.is_ntt);
                    (request.is_evaluation() == Some(descriptor.is_ntt) &&
                        used.insert(request.slot_key()))
                    .then_some((*identity, request))
                })
            })
        };
        let selected = select(true, &mut used).or_else(|| select(false, &mut used));
        let (storage, request) = selected.ok_or_else(|| {
            format!(
                "accepted prepared storage has no slot for {}x{} level {} on device {}",
                descriptor.rows, descriptor.columns, descriptor.level, descriptor.device
            )
        })?;
        requests.entry(storage).or_default().push(request);
        bindings.insert(
            descriptor.binding,
            PreparedMatrixBinding {
                storage,
                request,
                identity: super::gpu_prepared_lowering::PreparedStorageBinding {
                    storage_id: request.slot_key().0,
                    slot_id: request.slot_key().1,
                    slot_index: request.slot_key().2,
                    context: inventory
                        .iter()
                        .find(|(identity, _)| **identity == storage)
                        .map(|(_, (_, storage))| storage.context_identity())
                        .ok_or("selected prepared storage disappeared")?,
                    basis: descriptor.level,
                },
            },
        );
    }
    let mut compact_bindings = Vec::with_capacity(compact_descriptors.len());
    for descriptor in compact_descriptors {
        let binding = select_prepared_compact(backend, &inventory, descriptor, &mut used)?;
        for (storage, request) in binding.resources {
            requests.entry(storage).or_default().push(request);
        }
        compact_bindings.push(binding);
    }
    let (region, region_storages) =
        backend.reserve_standalone_prepared_region(&requests).map_err(|error| error.to_string())?;
    let storages =
        region_storages.into_iter().map(|(identity, (_, storage))| (identity, storage)).collect();
    Ok((region, storages, bindings, compact_bindings))
}

fn allocate_prepared_matrix(
    storages: &BTreeMap<u64, Arc<GpuPreparedStorage>>,
    binding: PreparedMatrixBinding,
    params: &GpuDCRTPolyParams,
    rows: usize,
    columns: usize,
    level: usize,
    is_ntt: bool,
) -> Result<Arc<GpuDCRTPolyMatrix>, String> {
    let storage =
        storages.get(&binding.storage).ok_or("prepared region is missing selected storage")?;
    let dispatch = storage.reserve(std::slice::from_ref(&binding.request))?.enter(Vec::new())?;
    let matrix = Arc::new(GpuDCRTPolyMatrix::new_empty_with_state(
        params, rows, columns, level, is_ntt, None,
    ));
    drop(dispatch.finish()?);
    Ok(matrix)
}

fn record_instance_storage_bindings(
    program: &mut super::gpu_prepared_lowering::PreparedProgram,
    descriptors: &[PreparedMatrixDescriptor],
    bindings: &BTreeMap<super::gpu_prepared_lowering::PreparedBindingId, PreparedMatrixBinding>,
) {
    let instance_count = descriptors
        .iter()
        .map(|descriptor| descriptor.binding.instance)
        .max()
        .map_or(1, |instance| instance + 1);
    let mut instances = vec![BTreeMap::new(); instance_count];
    for descriptor in descriptors {
        if let Some(binding) = bindings.get(&descriptor.binding) {
            instances[descriptor.binding.instance].insert(descriptor.binding, binding.identity);
        }
    }
    program.instance_storage_bindings = instances.into_boxed_slice();
}

#[cfg(all(test, feature = "gpu-instrumentation"))]
#[test]
fn test_gpu_prepared_forbidden_counter_categories_are_live() {
    reset_prepared_gpu_work_counters();
    PREPARED_WORK_GATE.store(1, Ordering::Release);
    for counter in 0..9 {
        record_prepared_forbidden(counter);
    }
    record_provisioning_begin();
    record_provisioning_permit();
    record_provisioning_append();
    record_prepared_generic_fallback();
    mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(true);
    mxx_primitives::poly::dcrt::gpu::gpu_test_record_event_creation();
    mxx_primitives::poly::dcrt::gpu::gpu_test_record_stream_creation();
    mxx_primitives::poly::dcrt::gpu::gpu_test_record_native_validation();
    mxx_primitives::poly::dcrt::gpu::gpu_test_record_cuda_allocation();
    mxx_primitives::poly::dcrt::gpu::gpu_test_record_kernel_launch();
    mxx_primitives::poly::dcrt::gpu::gpu_test_record_measurement_launch();
    mxx_primitives::poly::dcrt::gpu::gpu_test_set_work_gate(false);
    PREPARED_WORK_GATE.store(0, Ordering::Release);
    let counters = prepared_gpu_work_counters();
    assert_eq!(counters.graph_traversals, 1);
    assert_eq!(counters.graph_hashes, 1);
    assert_eq!(counters.assignments, 1);
    assert_eq!(counters.admissions, 1);
    assert_eq!(counters.native_validations, 2);
    assert_eq!(counters.reservations, 1);
    assert_eq!(counters.leases, 1);
    assert_eq!(counters.project_allocations, 1);
    assert_eq!(counters.dynamic_events, 2);
    assert_eq!(counters.dynamic_streams, 1);
    assert_eq!(counters.measurement_launches, 1);
    assert_eq!(counters.production_kernels, 1);
    assert_eq!(counters.cuda_allocations, 1);
    assert_eq!(counters.provisioning_begins, 1);
    assert_eq!(counters.provisioning_permits, 1);
    assert_eq!(counters.provisioning_appends, 1);
    assert_eq!(counters.generic_fallbacks, 1);
}

#[cfg(test)]
mod output_sharing_tests {
    use super::*;

    fn empty_execution() -> PreparedGpuFleetExecution {
        let instance = || {
            Arc::new(FleetInstance {
                state: Mutex::new(FleetInstanceState {
                    commands: Vec::new().into_boxed_slice(),
                    replay_steps: Arc::from([]),
                    output_commands: Vec::new().into_boxed_slice(),
                    small_output_commands: Vec::new().into_boxed_slice(),
                    input_values: Vec::new().into_boxed_slice(),
                    scalar_inputs: Vec::new().into_boxed_slice(),
                    scalar_slots: Arc::from([]),
                    control_scratch: Vec::new().into_boxed_slice(),
                    control_results: Vec::new().into_boxed_slice(),
                    selection_results: Vec::new().into_boxed_slice(),
                    poisoned: false,
                }),
            })
        };
        let region = Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test());
        let pool = Arc::new(PreparedGpuSlotPool {
            instances: vec![instance(), instance()].into_boxed_slice(),
            region,
            free_mask: AtomicUsize::new(0b11),
            poisoned: AtomicUsize::new(0),
            retired: Mutex::new(Vec::new()),
        });
        PreparedGpuFleetExecution {
            pool,
            rows: 0,
            columns: 0,
            program: None,
            control_commands: Arc::from([]),
            control_output_indices: Arc::from([]),
            output_control_indices: Arc::from([]),
            scalar_input_max_words: Arc::from([]),
            runtime_input_descriptors: Arc::from([]),
            output_matrix_ordinals: Arc::from([]),
            output_family_members: Arc::from([]),
            output_descriptors: Arc::from([]),
            output_names: Arc::from([]),
            output_indices: Arc::new(BTreeMap::new()),
            scalar_output_commands: Arc::new(BTreeMap::new()),
            host_output_descriptors: Arc::from([]),
        }
    }

    #[test]
    fn test_gpu_prepared_full_word_instance_mask() {
        let execution = PreparedGpuFleetExecution::from_command_instances(
            (0..usize::BITS)
                .map(|_| (Box::new([]) as Box<[PreparedCommand]>, Box::new([]) as Box<[usize]>))
                .collect(),
            Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test()),
            0,
            0,
        );
        let outputs = (0..usize::BITS).map(|_| execution.run().unwrap()).collect::<Vec<_>>();
        assert!(matches!(execution.run(), Err(PreparedGpuRunError::Busy(_))));
        drop(outputs);
        execution.pool.reclaim_retired().unwrap();
        assert_eq!(execution.pool.free_mask.load(Ordering::Acquire), usize::MAX);
    }

    #[test]
    fn retained_outputs_keep_execution_storage_alive_and_bound_slots() {
        let execution = empty_execution();
        let pool = Arc::downgrade(&execution.pool);
        let first = execution.run().unwrap();
        let second = execution.run().unwrap();
        assert!(matches!(execution.run(), Err(PreparedGpuRunError::Busy(_))));
        first.wait_until_ready().unwrap();
        let first_matrix = first.materialize().unwrap();
        assert_eq!(first_matrix.size(), (0, 0));
        drop(first);
        let third = execution.run().expect("released output slot is reusable");
        drop(execution);
        second.wait_until_ready().unwrap();
        third.wait_until_ready().unwrap();
        assert!(pool.upgrade().is_some(), "live outputs must retain their executable pool");
        drop(second);
        drop(third);
        assert!(pool.upgrade().is_none(), "pool must retire after the final output lease");
    }

    #[cfg(feature = "gpu-instrumentation")]
    #[test]
    fn output_return_and_materialization_do_not_allocate_or_traverse() {
        let execution = empty_execution();
        reset_prepared_gpu_work_counters();
        begin_prepared_gpu_work_gate();
        let output = execution.run().unwrap();
        output.wait_until_ready().unwrap();
        drop(output.materialize().unwrap());
        end_prepared_gpu_work_gate();
        let counters = prepared_gpu_work_counters();
        assert_eq!(counters.graph_traversals, 0);
        assert_eq!(counters.graph_hashes, 0);
        assert_eq!(counters.assignments, 0);
        assert_eq!(counters.admissions, 0);
        assert_eq!(counters.native_validations, 0);
        assert_eq!(counters.reservations, 0);
        assert_eq!(counters.leases, 0);
        assert_eq!(counters.project_allocations, 0);
        assert_eq!(counters.dynamic_events, 0);
        assert_eq!(counters.dynamic_streams, 0);
        assert_eq!(counters.measurement_launches, 0);
        assert_eq!(counters.cuda_allocations, 0);
        assert_eq!(counters.provisioning_begins, 0);
        assert_eq!(counters.provisioning_permits, 0);
        assert_eq!(counters.provisioning_appends, 0);
        assert_eq!(counters.generic_fallbacks, 0);
        drop(output);
    }

    #[test]
    fn repeated_drop_and_recreate_reuses_slots_without_stale_output_leases() {
        let execution = empty_execution();
        for _ in 0..32 {
            let output = execution.run().unwrap();
            output.wait_until_ready().unwrap();
            drop(output);
        }
        execution.pool.reclaim_retired().unwrap();
        assert_eq!(execution.pool.free_mask.load(Ordering::Acquire), 3);
        assert_eq!(execution.pool.poisoned.load(Ordering::Acquire), 0);
    }
}
