//! Plan-time lowering of a validated root graph into bound physical GPU work.
//!
//! A frame owns its scratch allocations for the lifetime of the plan. Replays
//! may replace input owners only after the previous GPU and I/O work finishes;
//! output owners remain frame-local and must be copied before being returned.

use crate::{
    artifact::ArtifactKey,
    backend::{
        BoundStorage, GpuResidentValue, PolyMatrix, RuntimeValue,
        poly_gpu::{GpuDcrtBackend, PhysicalExport},
    },
    gpu_execution_plan::{
        ColumnRange, CompiledGpuOp, CompiledGpuProgram, FrozenGpuPlan, GpuBindingSource,
        GpuExecutionSiteKey, GpuHashResourceSpec, GpuImplementation, GpuImplementationRegistry,
        GpuLayout, GpuLoopChoice, GpuLoopSiteKey, GpuNodeChoice, GpuPlanContract,
        GpuPreparedWorkspaceKind, KernelArg, PhysicalEncoding, PhysicalPart, PhysicalValue,
        PhysicalValueId, PhysicalView, StorageRef, reserve_gpu_export_slots, scope_shape_class,
    },
    gpu_physical_control::{
        ControlReset, ExternalIoImport, ExternalIoLoop, PhysicalWave, allocate_real_value,
        allocate_return_integer_family_value, allocate_return_integer_value,
        emit_integer_operation, emit_real_operation, finite_loop_count, fixed_child_env,
        lower_control_node, lower_wave_family_artifact_export, pack_resident_family,
        static_family_member,
    },
    gpu_schedule::GpuColumnInterval,
    matrix::gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrix, GpuSmallMatrixOutputDescriptor},
    poly::{
        PolyParams,
        dcrt::{
            gpu::{
                GpuDeviceBytes, GpuDeviceSeed, GpuDynamicExportTable, GpuExportSlot,
                GpuExportStatus, GpuHashTagPart, GpuIndexedMatrixTable, GpuIntegerOperation,
                GpuPreimageAttempt, GpuPreimageStatus, GpuSignedValues, GpuSignedValuesEncoding,
            },
            gpu_real::{GpuDeviceReal, GpuRealOperation},
        },
    },
};
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::{ArtifactAvailability, ArtifactType, ManifestArtifact},
    concretize_wire_type,
    graph::FrozenGraphScopeId,
    node::{ConstantMatrix, HashTagComponent, HashVariant, MatrixBinaryOp, NodeKind},
    types::{CoefficientBoundDomain, ConcreteMatrixType, ConcreteWireType, Port, WireRef},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::RangeInclusive,
    sync::Arc,
};

pub(crate) struct ExportTemplate {
    pub name: String,
    pub index: Option<usize>,
    pub occurrence: u64,
    pub site: u32,
    pub slot: usize,
    pub fragment_index: usize,
    pub final_chunk: bool,
    pub artifact_type: ArtifactType,
    pub availability: ArtifactAvailability,
    pub export: Arc<PhysicalExport>,
}

/// A canonical artifact is requested only at its first dependent Graph
/// region. The owner already exists at plan time; execution fills it in place.
#[derive(Clone)]
pub(crate) enum ImportDestination {
    Matrix {
        owner: Arc<GpuDCRTPolyMatrix>,
        ty: ConcreteMatrixType,
    },
    Bounded {
        owner: Arc<GpuSmallMatrix>,
        ty: ConcreteWireType,
    },
    Signed {
        owner: Arc<GpuSignedValues>,
        ty: ConcreteWireType,
    },
    Bytes {
        owner: Arc<GpuDeviceBytes>,
        length: usize,
    },
    Trapdoor {
        public: (Arc<GpuDCRTPolyMatrix>, ConcreteMatrixType),
        secret: [(Arc<GpuDCRTPolyMatrix>, ConcreteMatrixType); 6],
    },
}

pub(crate) struct ImportTemplate {
    pub before_operation: u32,
    pub key: ArtifactKey,
    pub descriptor: ManifestArtifact,
    pub expected_type: ArtifactType,
    pub staged: bool,
    pub destination: PhysicalValueId,
    pub upload_owner: ImportDestination,
}

pub(crate) struct PhysicalFrame {
    pub program: CompiledGpuProgram,
    pub owners: BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    pub slots: Vec<Arc<GpuExportSlot>>,
    pub device: i32,
    pub input_ids: BTreeMap<String, PhysicalValueId>,
    pub integer_input_owners: BTreeMap<String, Arc<GpuSignedValues>>,
    pub real_input_owners: BTreeMap<String, Arc<GpuDeviceReal>>,
    pub bytes_input_owners: BTreeMap<String, Arc<GpuDeviceSeed>>,
    pub sample_seeds: Vec<SampleSeed>,
    pub hash_resources: BTreeMap<u32, GpuHashResourceSpec>,
    pub real_owners: Vec<Arc<GpuDeviceReal>>,
    pub real_output_owners: BTreeMap<PhysicalValueId, Arc<GpuDeviceReal>>,
    pub indexed_tables: Vec<IndexedMatrixTableReplay>,
    pub dynamic_export_resources: BTreeMap<u32, (Arc<GpuDynamicExportTable>, Arc<GpuExportStatus>)>,
    pub preimage_replays: Vec<PreimageReplay>,
    pub trapdoor_public_ids: BTreeMap<PhysicalValueId, PhysicalValueId>,
    pub output_ids: BTreeMap<String, PhysicalValueId>,
    pub export_templates: Vec<ExportTemplate>,
    pub import_templates: Vec<ImportTemplate>,
    pub control_resets: Vec<ControlReset>,
    pub waves: Vec<PhysicalWave>,
    pub external_io_loops: Vec<ExternalIoLoop>,
    pub external_io_imports: Vec<ExternalIoImport>,
}

pub(crate) struct SampleSeed {
    pub owner: Arc<GpuDeviceSeed>,
    pub site: [u8; 32],
}

pub(crate) struct PreimageReplay {
    pub attempt: Arc<GpuPreimageAttempt>,
    pub status: Arc<GpuPreimageStatus>,
    pub planned_max: u32,
}

pub(crate) struct IndexedMatrixTableReplay {
    pub resource_id: u32,
    pub candidates: Vec<(PhysicalValueId, u32)>,
    pub encoding: PhysicalEncoding,
    pub table: Arc<GpuIndexedMatrixTable>,
}

fn host_integer_values(
    value: &RuntimeValue,
    expected: &ConcreteWireType,
) -> Result<Option<Vec<BigInt>>, String> {
    match (value, expected) {
        (RuntimeValue::Int(value), ConcreteWireType::Int) => Ok(Some(vec![value.clone()])),
        (
            RuntimeValue::IndexedFamily { element_type, values },
            ConcreteWireType::IndexedFamily { element, count },
        ) if element.as_ref() == &ConcreteWireType::Int &&
            element_type == &ConcreteWireType::Int &&
            values.len() == *count =>
        {
            values
                .iter()
                .map(|value| match value {
                    RuntimeValue::Int(value) => Ok(value.clone()),
                    _ => Err("GPU integer family has a noninteger member".to_owned()),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        }
        (RuntimeValue::Int(_), _) => {
            Err("GPU host integer input disagrees with its declared type".into())
        }
        (RuntimeValue::IndexedFamily { .. }, ConcreteWireType::IndexedFamily { element, .. })
            if element.as_ref() == &ConcreteWireType::Int =>
        {
            Err("GPU host integer family disagrees with its declared type".into())
        }
        _ => Ok(None),
    }
}

fn planned_integer_input(
    backend: &GpuDcrtBackend,
    device: i32,
    ty: &ConcreteWireType,
    values: &[BigInt],
    range: &RangeInclusive<BigInt>,
    storage: StorageRef,
) -> Result<(PhysicalValue, Arc<GpuResidentValue>, Arc<GpuSignedValues>), String> {
    if range.start() > range.end() ||
        values.is_empty() ||
        values.iter().any(|value| !range.contains(value))
    {
        return Err("GPU integer input is outside its frozen declared range".into());
    }
    let words = usize::try_from(range.start().bits().max(range.end().bits()).div_ceil(64))
        .map_err(|_| "GPU integer range width exceeds usize".to_owned())?
        .max(1);
    let params = backend.control_parameters_on_device(device)?;
    let owner = Arc::new(
        GpuSignedValues::from_bigints_with_words(&params, device, values, words)
            .map_err(|error| error.to_string())?,
    );
    owner.wait_until_ready().map_err(|error| error.to_string())?;
    let words_per_value = words.checked_add(1).ok_or("GPU integer word width overflows")?;
    let stride = u64::try_from(words_per_value)
        .ok()
        .and_then(|words| words.checked_mul(8))
        .ok_or("GPU integer byte stride overflows")?;
    let (origin, extent, byte_strides): (Box<[u64]>, Box<[u64]>, Box<[u64]>) = match ty {
        ConcreteWireType::Int if values.len() == 1 => {
            (Box::new([0, 0]), Box::new([1, words_per_value as u64]), Box::new([stride, 8]))
        }
        ConcreteWireType::IndexedFamily { element, count }
            if element.as_ref() == &ConcreteWireType::Int && *count == values.len() =>
        {
            (
                Box::new([0, 0, 0]),
                Box::new([*count as u64, 1, words_per_value as u64]),
                Box::new([stride, stride, 8]),
            )
        }
        _ => return Err("GPU integer input has an unsupported physical shape".into()),
    };
    let physical = PhysicalValue {
        ty: ty.clone(),
        encodings: Box::new([PhysicalEncoding::Signed(GpuSignedValuesEncoding::SignedWords(
            words,
        ))]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device,
            view: PhysicalView { byte_offset: 0, origin, extent, byte_strides, element_bytes: 8 },
        }]),
        integer_ranges: BTreeMap::from([(0, range.clone())]),
    };
    let bound = BoundStorage::from_signed_values(Arc::clone(&owner))?;
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, bound)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    Ok((physical, Arc::new(resident), owner))
}

fn planned_real_input(
    backend: &GpuDcrtBackend,
    device: i32,
    ty: &ConcreteWireType,
    value: f64,
    storage: StorageRef,
) -> Result<(PhysicalValue, Arc<GpuResidentValue>, Arc<GpuDeviceReal>), String> {
    if !matches!(ty, ConcreteWireType::Real | ConcreteWireType::ConstantReal) || !value.is_finite()
    {
        return Err("GPU real input has the wrong type or a nonfinite value".into());
    }
    let params = backend.control_parameters_on_device(device)?;
    let native = Arc::new(GpuDeviceReal::new(&params, device).map_err(|error| error.to_string())?);
    native.upload_f64(value).map_err(|error| error.to_string())?;
    let physical = PhysicalValue {
        ty: ty.clone(),
        encodings: Box::new([PhysicalEncoding::RealF64]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([1]),
                byte_strides: Box::new([8]),
                element_bytes: 8,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_device_real(Arc::clone(&native))?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    Ok((physical, Arc::new(resident), native))
}

pub(super) fn root_matrix_operation_identity(
    kind: &NodeKind,
    arguments: &[ConcreteWireType],
    output: &ConcreteWireType,
    bindings: &ParamEnv,
) -> Result<[u8; 32], String> {
    #[derive(Serialize)]
    struct Identity<'a> {
        kind: &'a NodeKind,
        concrete_argument_types: &'a [ConcreteWireType],
        concrete_output_types: &'a [ConcreteWireType],
        bindings: &'a ParamEnv,
    }
    let NodeKind::MatrixBinary(operation) = kind else {
        return Err("GPU root identity requires a matrix binary operation".into());
    };
    let [left, right] = arguments else {
        return Err("GPU root matrix identity requires two arguments".into());
    };
    let mut normalized = [left.clone(), right.clone()];
    let mut output = output.clone();
    let one_column = |value: &mut ConcreteWireType| -> Result<(), String> {
        let ConcreteWireType::Matrix(matrix) = value else {
            return Err("GPU root matrix identity requires full matrices".into());
        };
        matrix.columns = 1;
        Ok(())
    };
    match operation {
        MatrixBinaryOp::Add | MatrixBinaryOp::Subtract => {
            normalized.iter_mut().try_for_each(one_column)?;
        }
        MatrixBinaryOp::Multiply => {
            let left = left.matrix_type().ok_or("GPU multiply left type is absent")?;
            let right = right.matrix_type().ok_or("GPU multiply right type is absent")?;
            let scale_left =
                (right.rows, right.columns) == (1, 1) && (left.rows, left.columns) != (1, 1);
            one_column(&mut normalized[usize::from(!scale_left)])?;
        }
    }
    one_column(&mut output)?;
    let shape_bindings = ParamEnv {
        integers: bindings.integers.clone(),
        reals: bindings.reals.clone(),
        loop_indices: Default::default(),
    };
    mxx_ir_core::encoding::hash_canonical(&Identity {
        kind,
        concrete_argument_types: &normalized,
        concrete_output_types: &[output],
        bindings: &shape_bindings,
    })
    .map_err(|error| error.to_string())
}

/// Contiguous column blocks of nearly equal width, one per device that gets
/// any columns.
fn balanced_owner_intervals(columns: usize, devices: usize) -> Vec<GpuColumnInterval> {
    let mut start = 0;
    (0..devices)
        .filter_map(|device| {
            let length = columns / devices + usize::from(device < columns % devices);
            let interval = GpuColumnInterval { device, start, end: start + length };
            start += length;
            (length > 0).then_some(interval)
        })
        .collect()
}

fn append_matrix_candidate(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    env: &ParamEnv,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    types: &BTreeMap<WireRef, ConcreteWireType>,
    columns_per_job: usize,
    devices: usize,
    instance_class: u64,
    loop_site: Option<GpuLoopSiteKey>,
    layouts: &mut Vec<GpuLayout>,
    nodes: &mut Vec<GpuNodeChoice>,
) -> Result<(), String> {
    let NodeKind::MatrixBinary(kind) = node.kind() else {
        return Err("GPU candidate expected a matrix binary node".into());
    };
    let scope = validated
        .source
        .scope(scope_id)
        .ok_or_else(|| "GPU candidate scope is missing".to_owned())?;
    let wire = WireRef { node: node_id, port: Port(0) };
    let output_type = types
        .get(&wire)
        .ok_or_else(|| "GPU candidate output has no concrete type".to_owned())?
        .clone();
    let matrix = output_type
        .matrix_type()
        .ok_or_else(|| "GPU candidate output is not a matrix".to_owned())?;
    let argument_types = scope
        .arguments(node)
        .ok_or_else(|| "GPU candidate arguments are outside their scope".to_owned())?
        .iter()
        .map(|wire| {
            types
                .get(wire)
                .cloned()
                .ok_or_else(|| "GPU candidate argument has no concrete type".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let identity = root_matrix_operation_identity(node.kind(), &argument_types, &output_type, env)?;
    let layout_id =
        u32::try_from(layouts.len()).map_err(|_| "too many GPU candidate layouts".to_owned())?;
    // A product is compute-bound, so its output columns split evenly over
    // every device, each shard computed next to copies of its operands. An
    // elementwise operation would spend more on those copies than it saves
    // and stays on the home device.
    let (owner_intervals, columns_per_job) = if *kind == MatrixBinaryOp::Multiply {
        (balanced_owner_intervals(matrix.columns, devices), vec![columns_per_job; devices])
    } else {
        (
            vec![GpuColumnInterval { device: 0, start: 0, end: matrix.columns }],
            vec![columns_per_job],
        )
    };
    layouts.push(GpuLayout {
        id: layout_id,
        rows: matrix.rows,
        columns: matrix.columns,
        ring_dimension: matrix.ring.ring_dimension() as usize,
        representation: format!("{output_type:?}"),
        instance_device_stride: 0,
        owner_intervals,
    });
    let shape_class = scope_shape_class(validated, scope_id).map_err(|error| error.to_string())?;
    nodes.push(GpuNodeChoice {
        key: GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class },
        loop_site,
        operation_identity: identity,
        output_layouts: vec![layout_id],
        columns_per_job,
        implementation_variant: "direct".into(),
        preimage_max_attempts: None,
    });
    Ok(())
}

fn append_preimage_candidate(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    types: &BTreeMap<WireRef, ConcreteWireType>,
    columns_per_job: usize,
    devices: usize,
    instance_class: u64,
    loop_site: Option<GpuLoopSiteKey>,
    layouts: &mut Vec<GpuLayout>,
    nodes: &mut Vec<GpuNodeChoice>,
) -> Result<(), String> {
    let wire = WireRef { node: node_id, port: Port(0) };
    let output = types
        .get(&wire)
        .ok_or_else(|| "GPU preimage candidate has no concrete output".to_owned())?;
    let ConcreteWireType::Preimage { matrix, .. } = output else {
        return Err("GPU preimage candidate has no bounded output type".into());
    };
    let layout_id =
        u32::try_from(layouts.len()).map_err(|_| "too many GPU preimage layouts".to_owned())?;
    layouts.push(GpuLayout {
        id: layout_id,
        rows: matrix.rows,
        columns: matrix.columns,
        ring_dimension: matrix.ring.ring_dimension() as usize,
        representation: format!("{output:?}"),
        instance_device_stride: 0,
        owner_intervals: balanced_owner_intervals(matrix.columns, devices),
    });
    let max_attempts = crate::env::gpu_preimage_max_tile_attempts()?;
    let shape_class = scope_shape_class(validated, scope_id).map_err(|error| error.to_string())?;
    nodes.push(GpuNodeChoice {
        key: GpuExecutionSiteKey { site: node_id.0, shape_class, instance_class },
        loop_site,
        operation_identity: mxx_ir_core::encoding::hash_canonical(&(
            "direct-preimage-v1",
            output,
            matrix.columns,
        ))
        .map_err(|error| error.to_string())?,
        output_layouts: vec![layout_id],
        columns_per_job: vec![columns_per_job; devices],
        implementation_variant: "direct".into(),
        preimage_max_attempts: Some(max_attempts),
    });
    Ok(())
}

fn append_fixed_child_candidates(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    env: &ParamEnv,
    columns_per_job: usize,
    devices: usize,
    wave_instances: usize,
    instance_class: u64,
    loop_site: Option<GpuLoopSiteKey>,
    layouts: &mut Vec<GpuLayout>,
    nodes: &mut Vec<GpuNodeChoice>,
    loops: &mut Vec<GpuLoopChoice>,
    visited: &mut BTreeMap<(FrozenGraphScopeId, u64), ParamEnv>,
) -> Result<(), String> {
    let visit_key = (scope_id.clone(), instance_class);
    if let Some(previous) = visited.get(&visit_key) {
        if previous != env {
            return Err(format!(
                "GPU lexical scope {scope_id:?} has multiple concrete bindings without an instance class"
            ));
        }
        return Ok(());
    }
    visited.insert(visit_key, env.clone());
    let scope = validated
        .source
        .scope(scope_id)
        .ok_or_else(|| "GPU child candidate scope is missing".to_owned())?;
    let mut types = BTreeMap::new();
    for (index, node) in scope.nodes().iter().enumerate() {
        let node_id = mxx_ir_core::types::NodeId(
            u64::try_from(index).map_err(|_| "GPU child scope has too many nodes".to_owned())?,
        );
        for (port, declared) in node.output_types().iter().enumerate() {
            let port = Port(
                u32::try_from(port).map_err(|_| "GPU child node has too many ports".to_owned())?,
            );
            let ty = concretize_wire_type(
                declared,
                env,
                scope_id,
                node_id,
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| error.to_string())?;
            types.insert(WireRef { node: node_id, port }, ty);
        }
    }
    for (index, node) in scope.nodes().iter().enumerate() {
        let node_id = mxx_ir_core::types::NodeId(
            u64::try_from(index).map_err(|_| "GPU child scope has too many nodes".to_owned())?,
        );
        match node.kind() {
            NodeKind::Input { .. } |
            NodeKind::ConstantMatrix { .. } |
            NodeKind::FamilyPack { .. } |
            NodeKind::FamilyGetStatic { .. } |
            NodeKind::FamilyGetDynamic { .. } |
            NodeKind::Concat { .. } |
            NodeKind::BitExtract { .. } |
            NodeKind::BoolToInt |
            NodeKind::ConstantBool(_) |
            NodeKind::ConstantInt(_) |
            NodeKind::ConstantReal(_) |
            NodeKind::EvaluateInt(_) |
            NodeKind::IntBinary(_) |
            NodeKind::IntCompare(_) |
            NodeKind::IntToReal |
            NodeKind::RealBinary(_) |
            NodeKind::RealSqrt |
            NodeKind::Select { .. } |
            NodeKind::Slice { .. } |
            NodeKind::MatrixNegate |
            NodeKind::MatrixMulAccumulate { .. } |
            NodeKind::MatrixMulSmallRhs |
            NodeKind::MatrixScale { .. } |
            NodeKind::RingAutomorphism { .. } |
            NodeKind::MultiplyMonomial |
            NodeKind::IntMatrixVectorProduct { .. } |
            NodeKind::ModulusSwitch { .. } |
            NodeKind::GadgetDecompose { .. } |
            NodeKind::ExtractCoefficient { .. } |
            NodeKind::LiftIntegerToConstantPolynomial { .. } |
            NodeKind::ThresholdDecode { .. } |
            NodeKind::PackPolynomialCoefficients { .. } |
            NodeKind::PolynomialValues { .. } |
            NodeKind::CenteredRoundDivide { .. } |
            NodeKind::RnsModUp { .. } |
            NodeKind::RnsModDown { .. } |
            NodeKind::BlockModSwitch { .. } |
            NodeKind::CrtRecompose { .. } |
            NodeKind::CenteredRebase { .. } |
            NodeKind::Transpose |
            NodeKind::Tensor |
            NodeKind::TrapdoorPublic |
            NodeKind::GadgetTrapdoor { .. } |
            NodeKind::ModulusReduce { .. } |
            NodeKind::PolynomialFromValues { .. } |
            NodeKind::UniformResidueSample { .. } |
            NodeKind::UniformIntervalSample { .. } |
            NodeKind::GaussianSample { .. } |
            NodeKind::HashSample { .. } |
            NodeKind::HashIntFamily { .. } |
            NodeKind::TrapdoorSample { .. } => {}
            NodeKind::PreimageSample { .. } => append_preimage_candidate(
                validated,
                scope_id,
                node_id,
                &types,
                columns_per_job,
                devices,
                instance_class,
                loop_site,
                layouts,
                nodes,
            )?,
            NodeKind::MatrixBinary(_) => append_matrix_candidate(
                validated,
                scope_id,
                env,
                node_id,
                node,
                &types,
                columns_per_job,
                devices,
                instance_class,
                loop_site,
                layouts,
                nodes,
            )?,
            NodeKind::SubgraphCall(call) => {
                let child_id = validated
                    .source
                    .child_scope_id(scope_id, node_id)
                    .ok_or_else(|| "GPU fixed call has no child scope".to_owned())?;
                let child_env = fixed_child_env(scope_id, node_id, env, &call.bindings, None)?;
                append_fixed_child_candidates(
                    validated,
                    &child_id,
                    &child_env,
                    columns_per_job,
                    devices,
                    wave_instances,
                    instance_class,
                    loop_site,
                    layouts,
                    nodes,
                    loops,
                    visited,
                )?;
            }
            NodeKind::ParallelLoop(_)
                if crate::gpu_physical_control::is_vectorized_scalar_loop(
                    &validated.source,
                    scope_id,
                    node_id,
                    node,
                ) =>
            {
                // All lanes lower at once; there is no wave choice to freeze.
            }
            NodeKind::ParallelLoop(loop_node) => {
                let count =
                    usize::try_from(finite_loop_count(scope_id, node_id, node.kind(), env)?)
                        .map_err(|_| "GPU loop count exceeds host address space".to_owned())?;
                if count == 0 {
                    // A zero-iteration loop has no body invocation, choice, or
                    // sampled/exported side effect to materialize.
                    continue;
                }
                if wave_instances == 0 {
                    return Err("GPU loop W must be positive".into());
                }
                // One candidate W is shared by every loop site and capped by
                // each site's own finite count.
                let site_wave = wave_instances.min(count);
                let key = GpuLoopSiteKey {
                    site: node_id.0,
                    shape_class: scope_shape_class(validated, scope_id)
                        .map_err(|error| error.to_string())?,
                    instance_class,
                };
                loops.push(GpuLoopChoice {
                    key,
                    loop_count: count,
                    wave_instances: site_wave,
                    tail_instances: count % site_wave,
                });
                let child_id = validated
                    .source
                    .child_scope_id(scope_id, node_id)
                    .ok_or_else(|| "GPU parallel loop has no child scope".to_owned())?;
                let child_env = fixed_child_env(
                    scope_id,
                    node_id,
                    env,
                    &loop_node.bindings,
                    Some(loop_node.index_slot),
                )?;
                append_fixed_child_candidates(
                    validated,
                    &child_id,
                    &child_env,
                    columns_per_job,
                    devices,
                    wave_instances,
                    instance_class,
                    Some(key),
                    layouts,
                    nodes,
                    loops,
                    visited,
                )?;
            }
            NodeKind::SequentialLoop(loop_node) => {
                let count = finite_loop_count(scope_id, node_id, node.kind(), env)?;
                if count == 0 {
                    continue;
                }
                let key = GpuLoopSiteKey {
                    site: node_id.0,
                    shape_class: scope_shape_class(validated, scope_id)
                        .map_err(|error| error.to_string())?,
                    instance_class,
                };
                let child_id = validated
                    .source
                    .child_scope_id(scope_id, node_id)
                    .ok_or_else(|| "GPU sequential loop has no child scope".to_owned())?;
                let child_env = fixed_child_env(
                    scope_id,
                    node_id,
                    env,
                    &loop_node.bindings,
                    Some(loop_node.index_slot),
                )?;
                for index in 1..count {
                    let mut instance_env = child_env.clone();
                    instance_env.loop_indices.insert(loop_node.index_slot, index.into());
                    let child = validated
                        .source
                        .scope(&child_id)
                        .ok_or_else(|| "GPU sequential body is missing".to_owned())?;
                    for (child_index, child_node) in child.nodes().iter().enumerate() {
                        let child_node_id = mxx_ir_core::types::NodeId(child_index as u64);
                        for declared in child_node.output_types() {
                            let first = concretize_wire_type(
                                declared,
                                &child_env,
                                &child_id,
                                child_node_id,
                                crate::openfhe_guard::gen_modulus_and_warmup,
                            )
                            .map_err(|error| error.to_string())?;
                            let current = concretize_wire_type(
                                declared,
                                &instance_env,
                                &child_id,
                                child_node_id,
                                crate::openfhe_guard::gen_modulus_and_warmup,
                            )
                            .map_err(|error| error.to_string())?;
                            if first != current {
                                return Err(
                                    "GPU sequential loop has instance-dependent physical types"
                                        .into(),
                                );
                            }
                        }
                    }
                }
                append_fixed_child_candidates(
                    validated,
                    &child_id,
                    &child_env,
                    columns_per_job,
                    devices,
                    wave_instances,
                    instance_class,
                    Some(key),
                    layouts,
                    nodes,
                    loops,
                    visited,
                )?;
            }
        }
    }
    Ok(())
}

/// Construct one frozen candidate for caller-selected W and C. The caller
/// scores feasible pairs only after their physical allocations succeed.
pub(crate) fn single_root_physical_plan(
    validated: &ValidatedGraph,
    contract: GpuPlanContract,
    columns_per_job: usize,
    wave_instances: usize,
) -> Result<FrozenGpuPlan, String> {
    if contract.logical_to_physical_devices.is_empty() ||
        columns_per_job == 0 ||
        wave_instances == 0
    {
        return Err("root physical candidate needs a device and positive W/C".into());
    }
    let devices = contract.logical_to_physical_devices.len();
    let mut layouts = Vec::new();
    let mut nodes = Vec::new();
    let mut loops = Vec::new();
    append_fixed_child_candidates(
        validated,
        &FrozenGraphScopeId::Root,
        &validated.bindings,
        columns_per_job,
        devices,
        wave_instances,
        0,
        None,
        &mut layouts,
        &mut nodes,
        &mut loops,
        &mut BTreeMap::new(),
    )?;
    // Each choice freezes one tile width per logical device; a zero width
    // marks a device inactive for it, and unsharded nodes run on the home
    // device.
    for node in &mut nodes {
        node.columns_per_job.resize(devices, 0);
    }
    Ok(FrozenGpuPlan { contract, layouts, loops, nodes })
}

impl PhysicalFrame {
    /// Bind a new call's resident inputs without changing the frozen layout.
    /// The caller must establish the previous launch and I/O completion gate.
    pub(crate) fn rebind_inputs(
        &mut self,
        inputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<(), String> {
        if inputs.len() != self.input_ids.len() {
            return Err("GPU input set differs from the physical plan".into());
        }
        // Plan values viewing a previous input (slices, members, lane views)
        // share its allocations; they are re-derived over the new input.
        let mut replaced = std::collections::HashMap::new();
        let mut ready = Vec::new();
        for (name, id) in &self.input_ids {
            let value = inputs.get(name).ok_or_else(|| format!("missing GPU input {name}"))?;
            let planned = self
                .program
                .values
                .get(id.0 as usize)
                .ok_or_else(|| format!("GPU input {name} has no physical plan"))?;
            if let Some(owner) = self.integer_input_owners.get(name) {
                let integers = host_integer_values(value, &planned.ty)?
                    .ok_or_else(|| format!("GPU integer input {name} changed representation"))?;
                let range = planned
                    .integer_ranges
                    .get(&0)
                    .ok_or_else(|| format!("GPU integer input {name} lacks its frozen range"))?;
                if integers.iter().any(|integer| !range.contains(integer)) {
                    return Err(format!("GPU integer input {name} exceeds its frozen range"));
                }
                owner.upload_bigints(&integers).map_err(|error| error.to_string())?;
                owner.wait_until_ready().map_err(|error| error.to_string())?;
                continue;
            }
            if let Some(owner) = self.real_input_owners.get(name) {
                let RuntimeValue::Real(real) = value else {
                    return Err(format!("GPU real input {name} changed representation"));
                };
                owner.upload_f64(*real).map_err(|error| error.to_string())?;
                continue;
            }
            if self.bytes_input_owners.contains_key(name) {
                let RuntimeValue::Bytes(bytes) = value else {
                    return Err(format!("GPU bytes input {name} changed representation"));
                };
                if planned.ty != (ConcreteWireType::Bytes { length: 32 }) || bytes.len() != 32 {
                    return Err(format!("GPU bytes input {name} changed its frozen width"));
                }
                continue;
            }
            let device = planned
                .parts
                .first()
                .ok_or_else(|| format!("GPU input {name} has no physical part"))?
                .device;
            let resident = resident_input_with_type(value, &planned.ty, device)?;
            let resident =
                with_declared_integer_range(resident, &planned.ty, planned.integer_ranges.get(&0))?;
            let resident = with_planned_storage(resident, planned)?;
            if planned != resident.physical().as_ref() {
                return Err(format!("GPU input {name} changed its physical layout"));
            }
            if let Some(previous) = self.owners.insert(*id, Arc::clone(&resident)) {
                // Only allocations that actually change are redirected; the
                // same input bound again leaves every view untouched.
                let mut changed = false;
                for (slot, bound) in previous.storages() {
                    let replacement = resident
                        .storage(*slot)
                        .ok_or_else(|| format!("GPU input {name} lost a storage slot"))?;
                    if !Arc::ptr_eq(&bound.owner, &replacement.owner) {
                        replaced
                            .insert(Arc::as_ptr(&bound.owner).cast::<()>(), replacement.clone());
                        changed = true;
                    }
                }
                if changed {
                    ready.extend(resident.ready_events().iter().cloned());
                }
            }
        }
        if !replaced.is_empty() {
            let inputs = self.input_ids.values().copied().collect::<BTreeSet<_>>();
            for (id, owner) in self.owners.iter_mut() {
                if inputs.contains(id) {
                    continue;
                }
                if let Some(rebound) = owner.rebound(&replaced, &ready).map_err(str::to_owned)? {
                    *owner = Arc::new(rebound);
                }
            }
        }
        Ok(())
    }

    /// Keep plan-owned output storage bound across replays. The public API
    /// borrows results from this plan, so a subsequent execute may overwrite
    /// these exact owners after the previous GPU/I/O completion gate.
    pub(crate) fn bind_return_outputs(&mut self, _backend: &GpuDcrtBackend) -> Result<(), String> {
        for id in self.output_ids.values() {
            let planned =
                self.program.values.get(id.0 as usize).ok_or_else(|| {
                    "GPU return value is absent from the physical plan".to_owned()
                })?;
            let owner = self
                .owners
                .get(id)
                .ok_or_else(|| "GPU return value has no plan-owned storage".to_owned())?;
            if owner.physical().as_ref() != planned {
                return Err("GPU return value changed its physical layout".into());
            }
            if matches!(planned.ty, ConcreteWireType::Trapdoor { .. }) {
                if let Some(public_id) = self.trapdoor_public_ids.get(id) {
                    let public = self.owners.get(public_id).ok_or_else(|| {
                        "GPU trapdoor return has no paired public owner".to_owned()
                    })?;
                    if public.physical().ty.matrix_type() != planned.ty.matrix_type() {
                        return Err("GPU trapdoor return public type changed".into());
                    }
                }
            }
        }
        Ok(())
    }

    /// Call only after the compiled Graph completion event has succeeded.
    pub(crate) fn output_values(&self) -> Result<BTreeMap<String, RuntimeValue>, String> {
        self.output_ids
            .iter()
            .map(|(name, id)| {
                let resident = self
                    .owners
                    .get(id)
                    .ok_or_else(|| format!("GPU output {name} has no bound owner"))?;
                let ty = self
                    .program
                    .values
                    .get(id.0 as usize)
                    .ok_or_else(|| format!("GPU output {name} has no physical metadata"))?
                    .ty
                    .clone();
                let value = if !matches!(ty, ConcreteWireType::Matrix(_)) {
                    RuntimeValue::Resident(Arc::clone(resident))
                } else {
                    let matrix =
                        PolyMatrix::gpu(ty, Arc::clone(resident)).map_err(str::to_owned)?;
                    RuntimeValue::Matrix(matrix)
                };
                Ok((name.clone(), value))
            })
            .collect()
    }
}

pub(super) fn resident_input(value: &RuntimeValue) -> Result<Arc<GpuResidentValue>, String> {
    match value {
        RuntimeValue::Resident(owner) => Ok(Arc::clone(owner)),
        RuntimeValue::Matrix(matrix) => matrix
            .as_gpu()
            .cloned()
            .ok_or_else(|| "GPU physical input requires a resident matrix".into()),
        _ => Err("GPU physical input has no supported resident owner".into()),
    }
}

fn resident_input_with_type(
    value: &RuntimeValue,
    expected: &ConcreteWireType,
    device: i32,
) -> Result<Arc<GpuResidentValue>, String> {
    match value {
        RuntimeValue::IndexedFamily { element_type, values } => {
            let ConcreteWireType::IndexedFamily { element, count } = expected else {
                return Err("GPU resident family input has a nonfamily declared type".into());
            };
            if element.as_ref() != element_type || *count != values.len() {
                return Err("GPU resident family input has the wrong element type or count".into());
            }
            let members = values.iter().map(resident_input).collect::<Result<Vec<_>, _>>()?;
            pack_resident_family(expected.clone(), &members, device)
        }
        _ => resident_input(value),
    }
}

/// The same allocations under the planned storage slots. A resident value
/// produced by another plan numbers its storage with that plan's slots; each
/// of its slots maps to the one planned slot its parts occupy.
fn with_planned_storage(
    resident: Arc<GpuResidentValue>,
    planned: &PhysicalValue,
) -> Result<Arc<GpuResidentValue>, String> {
    let actual = resident.physical();
    if actual.parts.len() != planned.parts.len() {
        return Ok(resident);
    }
    let mut slots = BTreeMap::new();
    for (part, planned_part) in actual.parts.iter().zip(planned.parts.iter()) {
        if slots
            .insert(part.storage, planned_part.storage)
            .is_some_and(|slot| slot != planned_part.storage)
        {
            return Ok(resident);
        }
    }
    if slots.iter().all(|(actual, planned)| actual == planned) {
        return Ok(resident);
    }
    let storage = resident
        .storages()
        .map(|(slot, bound)| (slots.get(slot).copied().unwrap_or(*slot), bound.clone()))
        .collect::<BTreeMap<_, _>>();
    if storage.len() != resident.storages().count() {
        return Ok(resident);
    }
    let mut physical = actual.as_ref().clone();
    for (part, planned_part) in physical.parts.iter_mut().zip(planned.parts.iter()) {
        part.storage = planned_part.storage;
    }
    GpuResidentValue::new(Arc::new(physical), storage, resident.ready_events().into())
        .map(Arc::new)
        .map_err(str::to_owned)
}

fn with_declared_integer_range(
    resident: Arc<GpuResidentValue>,
    expected: &ConcreteWireType,
    range: Option<&RangeInclusive<BigInt>>,
) -> Result<Arc<GpuResidentValue>, String> {
    let integer = matches!(expected, ConcreteWireType::Int) ||
        matches!(expected, ConcreteWireType::IndexedFamily { element, .. }
            if element.as_ref() == &ConcreteWireType::Int);
    if !integer {
        return Ok(resident);
    }
    // A producer-guaranteed range is accepted when it lies within the declared
    // one; a value without a proven range needs a declared range.
    let proven = resident.physical().integer_ranges.get(&0).cloned();
    let range = match (range, &proven) {
        (Some(declared), Some(proven))
            if declared.start() > proven.start() || declared.end() < proven.end() =>
        {
            return Err("GPU resident integer input's proven range exceeds the declared one".into());
        }
        (Some(declared), _) => declared.clone(),
        (None, Some(proven)) => proven.clone(),
        (None, None) => return Err("GPU resident integer input has no declared range".into()),
    };
    if range.start() > range.end() {
        return Err("GPU resident integer input has an empty declared range".into());
    }
    let mut physical = resident.physical().as_ref().clone();
    physical.integer_ranges = BTreeMap::from([(0, range)]);
    resident.with_physical_view(Arc::new(physical)).map(Arc::new).map_err(str::to_owned)
}

pub(super) fn value_id(count: usize) -> Result<PhysicalValueId, String> {
    u32::try_from(count).map(PhysicalValueId).map_err(|_| "too many physical GPU values".into())
}

pub(super) fn matrix_column_view(
    values: &mut Vec<PhysicalValue>,
    owners: &mut BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    source: PhysicalValueId,
    range: ColumnRange,
) -> Result<PhysicalValueId, String> {
    let source_owner =
        owners.get(&source).ok_or_else(|| "GPU tile source has no bound owner".to_owned())?;
    let mut physical = values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU tile source has no physical metadata".to_owned())?
        .clone();
    let start =
        u64::try_from(range.start).map_err(|_| "GPU tile column index exceeds u64".to_owned())?;
    let end =
        u64::try_from(range.end).map_err(|_| "GPU tile column index exceeds u64".to_owned())?;
    if physical.parts.is_empty() {
        return Err("GPU column tile has no physical parts".into());
    }
    for part in physical.parts.iter_mut() {
        let bound = part.view.origin[1]
            .checked_add(part.view.extent[1])
            .ok_or_else(|| "GPU tile source column range overflows".to_owned())?;
        if start >= end || start < part.view.origin[1] || end > bound {
            return Err("GPU column tile is outside its source view".into());
        }
        let displacement = (start - part.view.origin[1])
            .checked_mul(part.view.byte_strides[1])
            .ok_or_else(|| "GPU column tile offset overflows".to_owned())?;
        part.view.byte_offset = part
            .view
            .byte_offset
            .checked_add(displacement)
            .ok_or_else(|| "GPU column tile address overflows".to_owned())?;
        part.view.origin[1] = start;
        part.view.extent[1] = end - start;
    }
    let view =
        source_owner.with_physical_view(Arc::new(physical.clone())).map_err(str::to_owned)?;
    let id = value_id(values.len())?;
    values.push(physical);
    owners.insert(id, Arc::new(view));
    Ok(id)
}

pub(super) fn predecessors_for(
    producer: &BTreeMap<PhysicalValueId, Vec<(ColumnRange, u32)>>,
    id: PhysicalValueId,
    needed: ColumnRange,
) -> Box<[u32]> {
    producer
        .get(&id)
        .into_iter()
        .flatten()
        .filter(|(written, _)| written.start < needed.end && needed.start < written.end)
        .map(|(_, op)| *op)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn all_predecessors(
    producer: &BTreeMap<PhysicalValueId, Vec<(ColumnRange, u32)>>,
    id: PhysicalValueId,
) -> Box<[u32]> {
    producer
        .get(&id)
        .into_iter()
        .flatten()
        .map(|(_, op)| *op)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(super) fn register_bindings(
    bindings: &mut Vec<GpuBindingSource>,
    values: &[PhysicalValue],
    id: PhysicalValueId,
) -> Result<u32, String> {
    let value = values
        .get(id.0 as usize)
        .ok_or_else(|| "GPU operation references an unknown physical value".to_owned())?;
    let depth = value
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU operation requires a matrix".to_owned())?
        .ring
        .crt_depth();
    if value.parts.len() != depth ||
        value.parts.iter().enumerate().any(|(index, part)| {
            part.leaf != 0 ||
                part.view.origin.len() != 4 ||
                part.view.origin[2] != index as u64 ||
                part.view.extent[2] != 1
        })
    {
        return Err("GPU operation needs one exact physical part per ordered CRT limb".into());
    }
    let base =
        u32::try_from(bindings.len()).map_err(|_| "too many GPU graph bindings".to_owned())?;
    for limb in 0..depth {
        bindings.push(GpuBindingSource::PhysicalPart {
            value: id,
            part: u32::try_from(limb).map_err(|_| "CRT part index exceeds u32".to_owned())?,
            limb: 0,
        });
    }
    Ok(base)
}

pub(super) fn physical_matrix(
    ty: &ConcreteMatrixType,
    encoding: PhysicalEncoding,
    storage: StorageRef,
    owner: Arc<crate::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix>,
) -> Result<(PhysicalValue, Arc<GpuResidentValue>), String> {
    let components = owner.binding_components().map_err(|error| error.to_string())?;
    let [component] = components.as_ref() else {
        return Err("GPU physical matrix needs an explicit multi-device shard plan".into());
    };
    let limbs = owner.binding_limbs().map_err(|error| error.to_string())?;
    let degree = usize::try_from(ty.ring.ring_dimension())
        .map_err(|_| "ring degree exceeds usize".to_owned())?;
    let depth = ty.ring.crt_depth();
    if component.limb_count != depth ||
        component.ring_dimension != degree ||
        limbs.len() != depth ||
        depth == 0 ||
        degree == 0
    {
        return Err("GPU allocation disagrees with the ordered CRT ring".into());
    }
    let mut parts = Vec::with_capacity(depth);
    for (index, limb) in limbs.iter().enumerate() {
        if limb.crt_limb_index != index ||
            limb.component_index != 0 ||
            limb.local_limb_index != index ||
            limb.physical_device != component.physical_device ||
            limb.data_bytes != component.data_bytes ||
            limb.poly_stride_bytes != component.bytes_per_poly ||
            limb.modulus != ty.ring.crt_moduli()[index] ||
            !matches!(limb.coefficient_bytes, 4 | 8) ||
            limb.data_address !=
                component
                    .data_address
                    .checked_add(limb.byte_offset as u64)
                    .ok_or_else(|| "GPU limb address overflows".to_owned())?
        {
            return Err("GPU allocation limb descriptor disagrees with its owner".into());
        }
        let row_stride = limb.row_stride_bytes;
        let limb_extent = degree
            .checked_mul(limb.coefficient_bytes)
            .ok_or_else(|| "GPU limb extent overflows".to_owned())?;
        parts.push(PhysicalPart {
            leaf: 0,
            storage,
            device: limb.physical_device,
            view: PhysicalView {
                byte_offset: limb.byte_offset as u64,
                origin: Box::new([0, 0, index as u64, 0]),
                extent: Box::new([ty.rows as u64, ty.columns as u64, 1, degree as u64]),
                byte_strides: Box::new([
                    row_stride as u64,
                    limb.poly_stride_bytes as u64,
                    limb_extent as u64,
                    limb.coefficient_bytes as u64,
                ]),
                element_bytes: limb.coefficient_bytes as u32,
            },
        });
    }
    let physical = PhysicalValue {
        ty: ConcreteWireType::Matrix(ty.clone()),
        encodings: Box::new([encoding]),
        parts: parts.into_boxed_slice(),
        integer_ranges: BTreeMap::new(),
    };
    let bound = BoundStorage::from_matrix_data(owner, 0)?;
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, bound)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    Ok((physical, Arc::new(resident)))
}

pub(super) struct PhysicalLoweringContext<'a> {
    pub validated: &'a ValidatedGraph,
    pub integer_input_ranges: &'a BTreeMap<String, RangeInclusive<BigInt>>,
    pub artifact_payload_sizes: &'a BTreeMap<ArtifactKey, usize>,
    pub backend: &'a GpuDcrtBackend,
    pub logical: &'a FrozenGpuPlan,
    pub device: i32,
    pub values: &'a mut Vec<PhysicalValue>,
    pub owners: &'a mut BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    pub wire_ids: &'a mut BTreeMap<WireRef, PhysicalValueId>,
    pub implementations: &'a mut GpuImplementationRegistry,
    pub operations: &'a mut Vec<CompiledGpuOp>,
    pub bindings: &'a mut Vec<GpuBindingSource>,
    pub producer: &'a mut BTreeMap<PhysicalValueId, Vec<(ColumnRange, u32)>>,
    pub family_member_producers:
        &'a mut BTreeMap<(PhysicalValueId, usize), Vec<(ColumnRange, u32)>>,
    pub control_resets: &'a mut Vec<ControlReset>,
    pub sample_seeds: &'a mut Vec<SampleSeed>,
    pub hash_resources: &'a mut BTreeMap<u32, GpuHashResourceSpec>,
    pub real_owners: &'a mut Vec<Arc<GpuDeviceReal>>,
    pub indexed_tables: &'a mut Vec<IndexedMatrixTableReplay>,
    pub dynamic_export_resources:
        &'a mut BTreeMap<u32, (Arc<GpuDynamicExportTable>, Arc<GpuExportStatus>)>,
    /// Device-owned loop index values in the current native control body.
    pub device_loop_indices: BTreeMap<u32, PhysicalValueId>,
    /// Lane count of a vectorized scalar loop body: each non-constant scalar
    /// holds one value per lane on its value-index axis. One outside it.
    pub lanes: usize,
    /// Current reusable parallel template, identified by site and static lane.
    /// Logical occurrence is supplied by the replay scheduler, not frozen here.
    pub active_parallel_template: Option<(GpuLoopSiteKey, usize)>,
    pub active_parallel_instances: Vec<(usize, ParamEnv)>,
    /// Lowering a device-resident body (conditional branch, retry or
    /// sequential loop) whose operations replay without the host, so a
    /// parallel loop inside runs every occurrence in one template.
    pub device_body: bool,
    pub preimage_replays: &'a mut Vec<PreimageReplay>,
    pub trapdoor_public_ids: &'a mut BTreeMap<PhysicalValueId, PhysicalValueId>,
    pub waves: &'a mut Vec<PhysicalWave>,
    pub import_templates: &'a mut Vec<ImportTemplate>,
    pub external_io_loops: &'a mut Vec<ExternalIoLoop>,
    pub external_io_imports: &'a mut Vec<ExternalIoImport>,
    pub crt_resource_next: &'a mut u32,
    /// A full CRT matrix value and its counterpart in the other full encoding.
    /// A conversion is emitted once per value and shared by later consumers
    /// of the same operation list; a body context starts its own table.
    pub converted: &'a mut BTreeMap<PhysicalValueId, PhysicalValueId>,
    /// The status word each device's direct integer operations report into.
    /// Errors are first-wins and each replay resets it once, instead of one
    /// host reset and readback per operation.
    pub integer_status: &'a mut BTreeMap<i32, PhysicalValueId>,
}

/// Reserve one reusable lane input for a selected artifact member. The caller
/// records an ImportTemplate for each wave occurrence, all referencing the
/// same coefficient destination; no family-wide allocation or read occurs.
pub(super) fn allocate_matrix_import_destination(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: &ConcreteMatrixType,
) -> Result<(PhysicalValueId, PhysicalValueId, Arc<GpuDCRTPolyMatrix>, u32), String> {
    let native =
        ctx.backend.allocate_physical_matrix(ty, ctx.device, PhysicalEncoding::FullCoeff)?;
    let storage = StorageRef::Input(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU import storages".to_owned())?,
    );
    let (physical, owner) =
        physical_matrix(ty, PhysicalEncoding::FullCoeff, storage, Arc::clone(&native))?;
    let coefficient = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(coefficient, owner);
    let eval_native =
        ctx.backend.allocate_physical_matrix(ty, ctx.device, PhysicalEncoding::FullEval)?;
    let eval_storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU import stagings".to_owned())?,
    );
    let (physical, owner) =
        physical_matrix(ty, PhysicalEncoding::FullEval, eval_storage, eval_native)?;
    let eval = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(eval, owner);
    let source_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let eval_binding = register_bindings(ctx.bindings, ctx.values, eval)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::ntt(false)).map_err(str::to_owned)?;
    let index =
        u32::try_from(ctx.operations.len()).map_err(|_| "too many GPU operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(eval),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(eval_binding),
        ]),
        outputs: Box::new([eval]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: Box::new([]),
        body: None,
    });
    ctx.producer.insert(eval, vec![(ColumnRange { start: 0, end: ty.columns }, index)]);
    Ok((coefficient, eval, native, index))
}

/// Allocate only the selected member's typed owner. Matrix imports insert a
/// coefficient-to-evaluation operation; all other encodings consume the owner
/// directly at the caller's first dependent operation.
pub(super) fn allocate_typed_import_destination(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: &ConcreteWireType,
    key: &ArtifactKey,
    payload_capacity: Option<usize>,
) -> Result<(PhysicalValueId, PhysicalValueId, ImportDestination, u32), String> {
    if let ConcreteWireType::Matrix(matrix) = ty {
        let (destination, graph_value, owner, before_operation) =
            allocate_matrix_import_destination(ctx, matrix)?;
        return Ok((
            destination,
            graph_value,
            ImportDestination::Matrix { owner, ty: matrix.clone() },
            before_operation,
        ));
    }
    if let ConcreteWireType::Trapdoor { matrix, .. } = ty {
        let (public_coeff, public_eval, public_owner, before_operation) =
            allocate_matrix_import_destination(ctx, matrix)?;
        let leaf_types = trapdoor_leaf_types(ty)?;
        let mut coefficient_leaves = Vec::with_capacity(6);
        let mut evaluation_leaves = Vec::with_capacity(6);
        let mut secret_owners = Vec::with_capacity(6);
        for leaf_ty in leaf_types {
            let (coefficient, evaluation, owner, _) =
                allocate_matrix_import_destination(ctx, &leaf_ty)?;
            coefficient_leaves.push(coefficient);
            evaluation_leaves.push(evaluation);
            secret_owners.push((owner, leaf_ty));
        }
        let leaves: [PhysicalValueId; 6] = evaluation_leaves
            .try_into()
            .map_err(|_| "GPU trapdoor import has the wrong leaf count")?;
        let secret = pack_trapdoor_leaves(ctx, ty.clone(), leaves, PhysicalEncoding::FullEval)?;
        ctx.trapdoor_public_ids.insert(secret, public_eval);
        let secret_owners: [(Arc<GpuDCRTPolyMatrix>, ConcreteMatrixType); 6] = secret_owners
            .try_into()
            .map_err(|_| "GPU trapdoor import has the wrong owner count")?;
        let destination = coefficient_leaves.into_iter().next().unwrap_or(public_coeff);
        return Ok((
            destination,
            secret,
            ImportDestination::Trapdoor {
                public: (public_owner, matrix.clone()),
                secret: secret_owners,
            },
            before_operation,
        ));
    }
    let before_operation =
        u32::try_from(ctx.operations.len()).map_err(|_| "too many GPU operations")?;
    let storage = StorageRef::Input(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU import storages")?,
    );
    let (physical, resident, upload_owner) = match ty {
        ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
            let (physical, resident, owner, _) =
                compact_value_owner(ctx.backend, ctx.device, ty.clone(), storage)?;
            (physical, resident, ImportDestination::Bounded { owner, ty: ty.clone() })
        }
        ConcreteWireType::Int => {
            let known_size = ctx.artifact_payload_sizes.get(key).copied();
            let payload_size = payload_capacity.or(known_size).ok_or_else(|| {
                "GPU Int artifact requires plan_with_store for payload size".to_owned()
            })?;
            if ctx.artifact_payload_sizes.is_empty() {
                return Err("GPU Int artifact requires plan_with_store for payload size".into());
            }
            if known_size.is_some_and(|known| payload_size < known) {
                return Err("GPU Int artifact capacity is smaller than its metadata size".into());
            }
            if payload_size == 0 {
                return Err("GPU Int artifact has no canonical signed byte capacity".into());
            }
            let bits = payload_size.checked_mul(8).ok_or("GPU Int artifact width overflows")?;
            let limit = BigInt::from(1u8) << (bits - 1);
            let range = -&limit..=limit - 1;
            let (physical, resident, owner) = planned_integer_input(
                ctx.backend,
                ctx.device,
                ty,
                &[BigInt::from(0u8)],
                &range,
                storage,
            )?;
            (physical, resident, ImportDestination::Signed { owner, ty: ty.clone() })
        }
        ConcreteWireType::Bytes { .. } | ConcreteWireType::TypedBlob { .. } => {
            let capacity = match ty {
                ConcreteWireType::Bytes { length } => *length,
                ConcreteWireType::TypedBlob { .. } => {
                    let known_size = ctx.artifact_payload_sizes.get(key).copied();
                    let payload_size = payload_capacity.or(known_size).ok_or(
                        "GPU TypedBlob artifact requires plan_with_store for payload size",
                    )?;
                    if ctx.artifact_payload_sizes.is_empty() {
                        return Err(
                            "GPU TypedBlob artifact requires plan_with_store for payload size"
                                .into(),
                        );
                    }
                    if known_size.is_some_and(|known| payload_size < known) {
                        return Err(
                            "GPU TypedBlob capacity is smaller than its metadata size".into()
                        );
                    }
                    payload_size
                        .checked_add(8)
                        .ok_or("GPU TypedBlob physical capacity overflows")?
                }
                _ => unreachable!(),
            };
            if capacity == 0 {
                return Err("GPU zero-length artifact bytes need an empty physical leaf".into());
            }
            let params = ctx.backend.control_parameters_on_device(ctx.device)?;
            let owner = Arc::new(
                GpuDeviceBytes::new(&params, ctx.device, capacity)
                    .map_err(|error| error.to_string())?,
            );
            let physical = PhysicalValue {
                ty: ty.clone(),
                encodings: Box::new([if matches!(ty, ConcreteWireType::Bytes { .. }) {
                    PhysicalEncoding::Bytes
                } else {
                    PhysicalEncoding::TypedBlobLengthPrefixed
                }]),
                parts: Box::new([PhysicalPart {
                    leaf: 0,
                    storage,
                    device: ctx.device,
                    view: PhysicalView {
                        byte_offset: 0,
                        origin: Box::new([0]),
                        extent: Box::new([capacity as u64]),
                        byte_strides: Box::new([1]),
                        element_bytes: 1,
                    },
                }]),
                integer_ranges: BTreeMap::new(),
            };
            let resident = GpuResidentValue::new(
                Arc::new(physical.clone()),
                BTreeMap::from([(storage, BoundStorage::from_device_bytes(Arc::clone(&owner))?)]),
                Box::new([]),
            )
            .map_err(str::to_owned)?;
            (physical, Arc::new(resident), ImportDestination::Bytes { owner, length: capacity })
        }
        _ => return Err("GPU selected artifact has no typed import destination".into()),
    };
    let destination = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(destination, resident);
    Ok((destination, destination, upload_owner, before_operation))
}

/// Copy `source` into new storage on `device` with the same layout, one
/// contiguous copy per storage after the source's writers. Each allocation
/// covers only the bytes the parts view, so a family member does not
/// replicate its whole family.
pub(super) fn replicate_to_device(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    device: i32,
) -> Result<PhysicalValueId, String> {
    let mut physical = ctx
        .values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU replica source has no physical metadata".to_owned())?
        .clone();
    let mut spans = BTreeMap::<StorageRef, (u64, u64)>::new();
    for part in physical.parts.iter() {
        let view = &part.view;
        let last = view
            .extent
            .iter()
            .zip(view.byte_strides.iter())
            .try_fold(u64::from(view.element_bytes), |end, (&extent, &stride)| {
                end.checked_add(extent.checked_sub(1)?.checked_mul(stride)?)
            })
            .and_then(|end| end.checked_add(view.byte_offset))
            .ok_or_else(|| "GPU replica part exceeds its storage".to_owned())?;
        let span = spans.entry(part.storage).or_insert((view.byte_offset, last));
        *span = (span.0.min(view.byte_offset), span.1.max(last));
    }
    let parameters = ctx.backend.control_parameters_on_device(device)?;
    let mut storage = BTreeMap::new();
    for (&slot, &(start, end)) in &spans {
        let bytes = usize::try_from(end - start)
            .map_err(|_| "GPU replica exceeds host address space".to_owned())?;
        let copy =
            GpuDeviceBytes::new(&parameters, device, bytes).map_err(|error| error.to_string())?;
        storage.insert(slot, BoundStorage::from_device_bytes(Arc::new(copy))?);
    }
    for part in physical.parts.iter_mut() {
        part.device = device;
        part.view.byte_offset -= spans[&part.storage].0;
    }
    let owner = GpuResidentValue::new(Arc::new(physical.clone()), storage, Box::new([]))
        .map_err(str::to_owned)?;
    let replica = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(replica, Arc::new(owner));
    let implementation =
        ctx.implementations.register(GpuImplementation::copy()).map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, source);
    let mut copies = Vec::new();
    for (slot, (start, end)) in spans {
        // The part that starts the span anchors its one contiguous copy.
        let part = ctx.values[source.0 as usize]
            .parts
            .iter()
            .position(|part| part.storage == slot && part.view.byte_offset == start)
            .and_then(|part| u32::try_from(part).ok())
            .ok_or_else(|| "GPU replica span has no anchoring part".to_owned())?;
        let binding = u32::try_from(ctx.bindings.len())
            .map_err(|_| "too many GPU graph bindings".to_owned())?;
        ctx.bindings.push(GpuBindingSource::PhysicalPart { value: source, part, limb: 0 });
        ctx.bindings.push(GpuBindingSource::PhysicalPart { value: replica, part, limb: 0 });
        let op = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(source),
                KernelArg::U32(part),
                KernelArg::Value(replica),
                KernelArg::U32(part),
                KernelArg::U64(end - start),
                KernelArg::U32(binding),
                KernelArg::U32(binding + 1),
            ]),
            outputs: Box::new([replica]),
            device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors: predecessors.clone(),
            body: None,
        });
        copies.push(op);
    }
    let columns = ctx.values[replica.0 as usize].ty.matrix_type().map_or(1, |ty| ty.columns);
    let whole = ColumnRange { start: 0, end: columns };
    ctx.producer.insert(replica, copies.into_iter().map(|copy| (whole, copy)).collect());
    Ok(replica)
}

/// Allocate a full-Eval matrix on `device` with its physical metadata.
fn allocate_device_matrix(
    backend: &GpuDcrtBackend,
    values: &mut Vec<PhysicalValue>,
    owners: &mut BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    ty: &ConcreteMatrixType,
    device: i32,
) -> Result<PhysicalValueId, String> {
    let owner = backend.allocate_physical_matrix(ty, device, PhysicalEncoding::FullEval)?;
    let storage = StorageRef::Scratch(
        u32::try_from(values.len()).map_err(|_| "too many GPU scratch storages".to_owned())?,
    );
    let (physical, resident) = physical_matrix(ty, PhysicalEncoding::FullEval, storage, owner)?;
    let id = value_id(values.len())?;
    values.push(physical);
    owners.insert(id, resident);
    Ok(id)
}

/// Copy the equal-shape matrix view `source` into `destination` on `device`
/// after `predecessors`, recording the copy as the destination's writer.
fn copy_matrix_view(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    destination: PhysicalValueId,
    predecessors: Box<[u32]>,
) -> Result<u32, String> {
    let source_binding = register_bindings(ctx.bindings, ctx.values, source)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, destination)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::matrix_copy_view())
        .map_err(str::to_owned)?;
    let index =
        u32::try_from(ctx.operations.len()).map_err(|_| "too many GPU operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source),
            KernelArg::U32(0),
            KernelArg::Value(destination),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(destination_binding),
        ]),
        outputs: Box::new([destination]),
        device: ctx.values[source.0 as usize].parts[0].device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    Ok(index)
}

/// The `range` columns of a full-Eval matrix as a new matrix on `device`: a
/// column window is first packed into a dense matrix on its own device, and
/// the dense matrix then moves in one contiguous copy.
fn matrix_columns_on_device(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    range: ColumnRange,
    device: i32,
) -> Result<PhysicalValueId, String> {
    let ty = ctx.values[source.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU device tile source is not a matrix".to_owned())?
        .clone();
    let dense = if range.start == 0 && range.end == ty.columns {
        source
    } else {
        let window = matrix_column_view(ctx.values, ctx.owners, source, range)?;
        let home = ctx.values[source.0 as usize].parts[0].device;
        let ty = ConcreteMatrixType { columns: range.end - range.start, ..ty };
        let dense = allocate_device_matrix(ctx.backend, ctx.values, ctx.owners, &ty, home)?;
        let predecessors = predecessors_for(ctx.producer, source, range);
        let copy = copy_matrix_view(ctx, window, dense, predecessors)?;
        ctx.producer.insert(dense, vec![(ColumnRange { start: 0, end: ty.columns }, copy)]);
        dense
    };
    replicate_to_device(ctx, dense, device)
}

pub(super) fn lower_matrix_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &mxx_ir_core::graph::GraphScope,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
    choice: &GpuNodeChoice,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU matrix arguments are outside the root scope".to_owned())?;
    let [left_wire, right_wire] = arguments.as_slice() else {
        return Err("GPU matrix operation has the wrong arity".into());
    };
    let left = *ctx
        .wire_ids
        .get(left_wire)
        .ok_or_else(|| "GPU left argument has no physical value".to_owned())?;
    let right = *ctx
        .wire_ids
        .get(right_wire)
        .ok_or_else(|| "GPU right argument has no physical value".to_owned())?;
    let left = full_eval_value(ctx, left)?;
    let right = full_eval_value(ctx, right)?;
    let PhysicalLoweringContext {
        backend,
        logical,
        device,
        values,
        owners,
        wire_ids,
        implementations,
        ..
    } = ctx;
    let device = *device;
    let NodeKind::MatrixBinary(kind) = node.kind() else {
        return Err("physical matrix lowerer received a nonmatrix node".into());
    };
    let wire = WireRef { node: node_id, port: Port(0) };
    if node.output_types().len() != 1 {
        return Err("GPU matrix operation has multiple outputs".into());
    }
    let left_ty = values[left.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU left operand is not a matrix".to_owned())?;
    let right_ty = values[right.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU right operand is not a matrix".to_owned())?;
    let output_ty = wire_types
        .get(&wire)
        .and_then(ConcreteWireType::matrix_type)
        .ok_or_else(|| "GPU matrix output has no validated type".to_owned())?;
    if left_ty.ring != output_ty.ring ||
        right_ty.ring != output_ty.ring ||
        values[left.0 as usize].encodings.as_ref() != [PhysicalEncoding::FullEval] ||
        values[right.0 as usize].encodings.as_ref() != [PhysicalEncoding::FullEval]
    {
        return Err("GPU matrix operands need the exact ordered Eval CRT basis".into());
    }
    let implementation = match kind {
        MatrixBinaryOp::Add => GpuImplementation::matrix_add_sub(false),
        MatrixBinaryOp::Subtract => GpuImplementation::matrix_add_sub(true),
        MatrixBinaryOp::Multiply => GpuImplementation::matrix_mul(false),
    };
    let implementation = implementations.register(implementation).map_err(str::to_owned)?;
    let owner = backend.allocate_physical_matrix(output_ty, device, PhysicalEncoding::FullEval)?;
    let storage = StorageRef::Scratch(
        u32::try_from(values.len()).map_err(|_| "too many GPU scratch storages".to_owned())?,
    );
    let (physical, resident) =
        physical_matrix(output_ty, PhysicalEncoding::FullEval, storage, owner)?;
    let output = value_id(values.len())?;
    values.push(physical);
    owners.insert(output, resident);
    wire_ids.insert(wire, output);
    if choice.output_layouts.len() != 1 {
        return Err("GPU matrix choice differs from the validated operation".into());
    }
    let layout = logical
        .layouts
        .iter()
        .find(|layout| layout.id == choice.output_layouts[0])
        .ok_or_else(|| "GPU output layout is absent from the frozen plan".to_owned())?;
    if layout.rows != output_ty.rows ||
        layout.columns != output_ty.columns ||
        layout.ring_dimension != output_ty.ring.ring_dimension() as usize ||
        choice.columns_per_job.iter().all(|width| *width == 0)
    {
        return Err("GPU output layout or column width differs from the frozen plan".into());
    }
    let schedule =
        layout.schedule(&choice.columns_per_job, 0).map_err(|error| error.to_string())?;
    let left_columns = values[left.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU matrix left operand is not a matrix".to_owned())?
        .columns;
    let devices = logical
        .contract
        .logical_to_physical_devices
        .iter()
        .map(|&device| i32::try_from(device).map_err(|_| "GPU device ID overflows".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    if schedule.waves().flatten().any(|job| job.device >= devices.len()) {
        return Err("GPU column job names an unknown device".into());
    }
    let mut tile_producers = Vec::new();
    // Operand columns already copied to a device, shared by its later jobs.
    let mut device_tiles = BTreeMap::new();
    for job in schedule.waves().flatten() {
        // A device-resident body replays on one device, and a lane already
        // running on another device keeps its product there.
        let job_device =
            if ctx.device_body || device != devices[0] { device } else { devices[job.device] };
        let output_range = ColumnRange { start: job.start, end: job.end };
        // Output columns read the same columns of an elementwise operand, and
        // the whole left operand of a product.
        let left_range = match kind {
            MatrixBinaryOp::Add | MatrixBinaryOp::Subtract => output_range,
            MatrixBinaryOp::Multiply => ColumnRange { start: 0, end: left_columns },
        };
        let right_range = output_range;
        let mut operand = |ctx: &mut PhysicalLoweringContext<'_>, id, range: ColumnRange| {
            if job_device == device {
                let tile = matrix_column_view(ctx.values, ctx.owners, id, range)?;
                return Ok::<_, String>((tile, predecessors_for(ctx.producer, id, range)));
            }
            let key = (id, range.start, range.end, job_device);
            let tile = match device_tiles.get(&key) {
                Some(&tile) => tile,
                None => {
                    let tile = matrix_columns_on_device(ctx, id, range, job_device)?;
                    device_tiles.insert(key, tile);
                    tile
                }
            };
            Ok((tile, all_predecessors(ctx.producer, tile)))
        };
        let (left_tile, left_predecessors) = operand(ctx, left, left_range)?;
        let (right_tile, right_predecessors) = operand(ctx, right, right_range)?;
        let output_tile = if job_device == device {
            matrix_column_view(ctx.values, ctx.owners, output, output_range)?
        } else {
            let ty = ConcreteMatrixType {
                columns: output_range.end - output_range.start,
                ..output_ty.clone()
            };
            allocate_device_matrix(ctx.backend, ctx.values, ctx.owners, &ty, job_device)?
        };
        let left_binding = register_bindings(ctx.bindings, ctx.values, left_tile)?;
        let right_binding = register_bindings(ctx.bindings, ctx.values, right_tile)?;
        let output_binding = register_bindings(ctx.bindings, ctx.values, output_tile)?;
        let predecessors = left_predecessors
            .iter()
            .chain(right_predecessors.iter())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let op_index = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(left_tile),
                KernelArg::U32(0),
                KernelArg::Value(right_tile),
                KernelArg::U32(0),
                KernelArg::Value(output_tile),
                KernelArg::U32(0),
                KernelArg::U32(left_binding),
                KernelArg::U32(right_binding),
                KernelArg::U32(output_binding),
            ]),
            outputs: Box::new([output_tile]),
            device: job_device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors,
            body: None,
        });
        if job_device == device {
            tile_producers.push((output_range, op_index));
            continue;
        }
        // Gather the shard home, then into the output's columns.
        let width = ColumnRange { start: 0, end: output_range.end - output_range.start };
        ctx.producer.insert(output_tile, vec![(width, op_index)]);
        let gathered = replicate_to_device(ctx, output_tile, device)?;
        let home_tile = matrix_column_view(ctx.values, ctx.owners, output, output_range)?;
        let predecessors = all_predecessors(ctx.producer, gathered);
        let copy = copy_matrix_view(ctx, gathered, home_tile, predecessors)?;
        tile_producers.push((output_range, copy));
    }
    if tile_producers.is_empty() {
        return Err("GPU matrix output has no planned column jobs".into());
    }
    ctx.producer.insert(output, tile_producers);
    Ok(())
}

pub(super) fn lower_zero_matrix_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    node_id: mxx_ir_core::types::NodeId,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let wire = WireRef { node: node_id, port: Port(0) };
    let ty = wire_types
        .get(&wire)
        .and_then(ConcreteWireType::matrix_type)
        .ok_or_else(|| "GPU zero output has no concrete matrix type".to_owned())?;
    let native =
        ctx.backend.allocate_physical_matrix(ty, ctx.device, PhysicalEncoding::FullEval)?;
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU scratch storages".to_owned())?,
    );
    let (physical, resident) = physical_matrix(ty, PhysicalEncoding::FullEval, storage, native)?;
    let bytes = resident
        .storage(storage)
        .ok_or_else(|| "GPU zero allocation is missing its storage".to_owned())?
        .bytes;
    let output = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(output, resident);
    ctx.wire_ids.insert(wire, output);
    let binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::zero()).map_err(str::to_owned)?;
    let index =
        u32::try_from(ctx.operations.len()).map_err(|_| "too many GPU operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::U64(bytes),
            KernelArg::U32(binding),
        ]),
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [0; 3],
        block: [0; 3],
        shared_bytes: 0,
        predecessors: Box::new([]),
        body: None,
    });
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: ty.columns }, index)]);
    Ok(())
}

/// A direct device sample uses a plan-owned seed. The execution nonce is
/// uploaded into that stable owner before each launch; no host matrix sample
/// or capture path participates in the operation.
pub(super) fn lower_sample_matrix_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let mut interval = (0i64, 0i64);
    let (implementation, sigma, max_bound, coefficient_modulus) = match node.kind() {
        NodeKind::UniformResidueSample { .. } => (GpuImplementation::sample(false), 0.0, 0, 1),
        NodeKind::UniformIntervalSample { range, .. } => {
            let minimum = range
                .minimum
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?;
            let maximum = range
                .maximum
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?;
            let (Some(minimum), Some(maximum)) = (minimum.to_i64(), maximum.to_i64()) else {
                return Err("GPU interval sample bounds exceed the native i64 sampler".into());
            };
            if minimum > maximum {
                return Err("GPU interval sample has an empty range".into());
            }
            interval = (minimum, maximum);
            (GpuImplementation::sample_interval(), 0.0, 0, 1)
        }
        NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
            let sigma = sigma
                .evaluate_f64_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?;
            let max_bound = max_coefficient_bound
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?
                .to_u64()
                .ok_or_else(|| "GPU Gaussian coefficient bound exceeds u64".to_owned())?;
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err("GPU Gaussian sigma must be finite and positive".into());
            }
            // Native raw Gaussian accepts Q=0 when the full CRT product is
            // wider than u64; an i64 draw then remains its centered value.
            let modulus = wire_types
                .get(&WireRef { node: node_id, port: Port(0) })
                .and_then(ConcreteWireType::matrix_type)
                .ok_or_else(|| "GPU Gaussian output lacks a matrix type".to_owned())?
                .ring
                .modulus()
                .to_u64()
                .unwrap_or(0);
            (GpuImplementation::sample(true), sigma, max_bound, modulus)
        }
        _ => return Err("GPU sample lowerer received a different node kind".into()),
    };
    let wire = WireRef { node: node_id, port: Port(0) };
    let ty = wire_types
        .get(&wire)
        .and_then(ConcreteWireType::matrix_type)
        .ok_or_else(|| "GPU sample output lacks a validated matrix type".to_owned())?;
    // The value ordinal distinguishes lanes cloned from one lexical node
    // within a single W-lane graph. It is frozen for every sequential replay.
    let site =
        mxx_ir_core::encoding::hash_canonical(&(scope_id, node_id, Port(0), ctx.values.len()))
            .map_err(|error| error.to_string())?;
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let seed_native =
        Arc::new(GpuDeviceSeed::new(&params, ctx.device).map_err(|error| error.to_string())?);
    seed_native.upload(&[0; 32]).map_err(|error| error.to_string())?;
    let seed_storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU sample seeds".to_owned())?,
    );
    let seed_physical = PhysicalValue {
        ty: ConcreteWireType::Bytes { length: 32 },
        encodings: Box::new([PhysicalEncoding::Bytes]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage: seed_storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([32]),
                byte_strides: Box::new([1]),
                element_bytes: 1,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let seed_resident = GpuResidentValue::new(
        Arc::new(seed_physical.clone()),
        BTreeMap::from([(seed_storage, BoundStorage::from_device_seed(Arc::clone(&seed_native))?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let seed_id = value_id(ctx.values.len())?;
    ctx.values.push(seed_physical);
    ctx.owners.insert(seed_id, Arc::new(seed_resident));
    ctx.sample_seeds.push(SampleSeed { owner: seed_native, site });

    let coefficient_native =
        ctx.backend.allocate_physical_matrix(ty, ctx.device, PhysicalEncoding::FullCoeff)?;
    let coefficient_storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len())
            .map_err(|_| "too many GPU sample coefficients".to_owned())?,
    );
    let (coefficient_physical, coefficient_owner) =
        physical_matrix(ty, PhysicalEncoding::FullCoeff, coefficient_storage, coefficient_native)?;
    let coefficient = value_id(ctx.values.len())?;
    ctx.values.push(coefficient_physical);
    ctx.owners.insert(coefficient, coefficient_owner);
    let evaluation_native =
        ctx.backend.allocate_physical_matrix(ty, ctx.device, PhysicalEncoding::FullEval)?;
    let evaluation_storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len())
            .map_err(|_| "too many GPU sample evaluations".to_owned())?,
    );
    let (evaluation_physical, evaluation_owner) =
        physical_matrix(ty, PhysicalEncoding::FullEval, evaluation_storage, evaluation_native)?;
    let evaluation = value_id(ctx.values.len())?;
    ctx.values.push(evaluation_physical);
    ctx.owners.insert(evaluation, evaluation_owner);
    ctx.wire_ids.insert(wire, evaluation);

    let sample_implementation =
        ctx.implementations.register(implementation).map_err(str::to_owned)?;
    let coefficient_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let seed_binding =
        u32::try_from(ctx.bindings.len()).map_err(|_| "too many GPU graph bindings".to_owned())?;
    ctx.bindings.push(GpuBindingSource::PhysicalPart { value: seed_id, part: 0, limb: 0 });
    let sample_index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU sample operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation: sample_implementation,
        arguments: Box::new([
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(seed_id),
            KernelArg::U32(0),
            KernelArg::F64(sigma),
            KernelArg::U64(max_bound),
            KernelArg::U64(coefficient_modulus),
            KernelArg::I64(interval.0),
            KernelArg::I64(interval.1),
            KernelArg::U64(ty.columns as u64),
            KernelArg::U64(0),
            KernelArg::U32(coefficient_binding),
            KernelArg::U32(seed_binding),
        ]),
        outputs: Box::new([coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: Box::new([]),
        body: None,
    });
    ctx.producer
        .insert(coefficient, vec![(ColumnRange { start: 0, end: ty.columns }, sample_index)]);
    let ntt_implementation =
        ctx.implementations.register(GpuImplementation::ntt(false)).map_err(str::to_owned)?;
    let evaluation_binding = register_bindings(ctx.bindings, ctx.values, evaluation)?;
    let ntt_index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU sample operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation: ntt_implementation,
        arguments: Box::new([
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(evaluation),
            KernelArg::U32(0),
            KernelArg::U32(coefficient_binding),
            KernelArg::U32(evaluation_binding),
        ]),
        outputs: Box::new([evaluation]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: Box::new([sample_index]),
        body: None,
    });
    ctx.producer.insert(evaluation, vec![(ColumnRange { start: 0, end: ty.columns }, ntt_index)]);
    Ok(())
}

/// Encode immutable IR coefficients in the one canonical coefficient codec.
/// This performs only static data preparation; every runtime matrix operation
/// remains in the direct GPU graph.
fn encode_static_matrix(
    ty: &ConcreteMatrixType,
    value: &ConstantMatrix,
    env: &ParamEnv,
    regular_gadget_digits_per_tower: Option<usize>,
) -> Result<Vec<u8>, String> {
    let degree = ty.ring.ring_dimension() as usize;
    let count = ty
        .rows
        .checked_mul(ty.columns)
        .and_then(|n| n.checked_mul(degree))
        .ok_or_else(|| "GPU constant matrix shape overflows".to_owned())?;
    let mut coefficients = vec![BigInt::from(0); count];
    let mut set_constant = |row: usize, column: usize, coefficient: BigInt| {
        coefficients[(row * ty.columns + column) * degree] = coefficient;
    };
    let evaluate = |expression: &mxx_ir_core::expr::IntExpr| {
        expression
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| error.to_string())
    };
    match value {
        ConstantMatrix::Zero => {}
        ConstantMatrix::Identity if ty.rows == ty.columns => {
            for index in 0..ty.rows {
                set_constant(index, index, BigInt::from(1));
            }
        }
        ConstantMatrix::UnitRow { index } if ty.rows == 1 => {
            let index = evaluate(index)?
                .to_usize()
                .ok_or_else(|| "GPU constant unit row index is invalid".to_owned())?;
            if index >= ty.columns {
                return Err("GPU constant unit row index is out of range".into());
            }
            set_constant(0, index, BigInt::from(1));
        }
        ConstantMatrix::UnitColumn { index } if ty.columns == 1 => {
            let index = evaluate(index)?
                .to_usize()
                .ok_or_else(|| "GPU constant unit column index is invalid".to_owned())?;
            if index >= ty.rows {
                return Err("GPU constant unit column index is out of range".into());
            }
            set_constant(index, 0, BigInt::from(1));
        }
        ConstantMatrix::Gadget { base, small }
            if ty.rows > 0 && ty.columns.is_multiple_of(ty.rows) =>
        {
            let base = evaluate(base)?;
            let digits = ty.columns / ty.rows;
            let values = if *small {
                (0..digits)
                    .map(|digit| {
                        u32::try_from(digit)
                            .map(|exponent| base.pow(exponent))
                            .map_err(|_| "GPU small gadget exponent exceeds u32".to_owned())
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                let moduli = ty.ring.crt_moduli();
                let per_tower = regular_gadget_digits_per_tower
                    .ok_or_else(|| "GPU regular gadget has no parameter layout".to_owned())?;
                if per_tower == 0 || digits % per_tower != 0 || digits / per_tower > moduli.len() {
                    return Err("GPU regular gadget digits do not span whole CRT towers".into());
                }
                let modulus = ty.ring.modulus();
                let mut values = Vec::with_capacity(digits);
                for &prime in moduli.iter().take(digits / per_tower) {
                    let prime_big = BigInt::from(prime);
                    let quotient = &modulus / &prime_big;
                    let quotient_mod = (&quotient % &prime_big)
                        .to_u64()
                        .ok_or_else(|| "GPU gadget CRT residue exceeds u64".to_owned())?;
                    let inverse = crate::utils::mod_inverse(quotient_mod, prime)
                        .ok_or_else(|| "GPU gadget CRT basis is not invertible".to_owned())?;
                    let idempotent = quotient * BigInt::from(inverse);
                    for digit in 0..per_tower {
                        let exponent = u32::try_from(digit)
                            .map_err(|_| "GPU gadget exponent exceeds u32".to_owned())?;
                        values.push((&idempotent * base.pow(exponent)) % &modulus);
                    }
                }
                values
            };
            for row in 0..ty.rows {
                for (digit, value) in values.iter().enumerate() {
                    set_constant(row, row * digits + digit, value.clone());
                }
            }
        }
        ConstantMatrix::PowerOfBase { base, exponent } if ty.rows == 1 && ty.columns == 1 => {
            let exponent = evaluate(exponent)?
                .to_u32()
                .ok_or_else(|| "GPU constant power exponent is invalid".to_owned())?;
            set_constant(0, 0, evaluate(base)?.pow(exponent));
        }
        ConstantMatrix::Rotation { exponent } if ty.rows == 1 && ty.columns == 1 => {
            let exponent = evaluate(exponent)?
                .to_usize()
                .ok_or_else(|| "GPU constant rotation exponent is invalid".to_owned())?;
            let position = exponent % (2 * degree);
            coefficients[position % degree] =
                if position < degree { BigInt::from(1) } else { BigInt::from(-1) };
        }
        ConstantMatrix::Polynomial { coefficients: declared }
            if ty.rows == 1 && ty.columns == 1 && declared.len() <= degree =>
        {
            for (index, expression) in declared.iter().enumerate() {
                coefficients[index] = evaluate(expression)?;
            }
        }
        _ => return Err("GPU static constant has an unsupported shape or gadget layout".into()),
    }
    let modulus = ty.ring.modulus();
    let half = &modulus / 2;
    let centered = coefficients
        .into_iter()
        .map(|value| {
            let residue = ((value % &modulus) + &modulus) % &modulus;
            if residue > half { (true, &modulus - residue) } else { (false, residue) }
        })
        .map(|(negative, magnitude)| {
            let magnitude = magnitude
                .to_biguint()
                .ok_or_else(|| "GPU constant has negative magnitude".to_owned())?;
            Ok::<_, String>((negative, magnitude))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let maximum_bits =
        centered.iter().map(|(_, magnitude)| magnitude.bits() as usize).max().unwrap_or(0);
    let bit_width = if maximum_bits == 0 { 0 } else { maximum_bits + 1 };
    let bit_width = u16::try_from(bit_width)
        .map_err(|_| "GPU constant coefficient bit width exceeds u16".to_owned())?;
    let total_bits = centered
        .len()
        .checked_mul(bit_width as usize)
        .ok_or_else(|| "GPU constant payload size overflows".to_owned())?;
    let mut payload = vec![0u8; total_bits.div_ceil(8)];
    for (index, (negative, magnitude)) in centered.iter().enumerate() {
        let bytes = magnitude.to_bytes_le();
        let base = index * bit_width as usize;
        for bit in 0..bit_width.saturating_sub(1) as usize {
            if bytes.get(bit / 8).is_some_and(|byte| byte & (1 << (bit % 8)) != 0) {
                let position = base + bit;
                payload[position / 8] |= 1 << (position % 8);
            }
        }
        if *negative {
            let position = base + bit_width as usize - 1;
            payload[position / 8] |= 1 << (position % 8);
        }
    }
    bincode::encode_to_vec(
        (
            1u8,
            0u8,
            u32::try_from(ty.ring.crt_depth() - 1)
                .map_err(|_| "GPU constant CRT depth exceeds u32".to_owned())?,
            ty.rows,
            ty.columns,
            bit_width,
            bit_width.div_ceil(8),
            payload,
        ),
        bincode::config::standard(),
    )
    .map_err(|error| error.to_string())
}

pub(super) fn lower_static_matrix_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let NodeKind::ConstantMatrix { value, .. } = node.kind() else {
        return Err("GPU static matrix lowerer received another node".into());
    };
    if matches!(value, ConstantMatrix::Zero) {
        return lower_zero_matrix_node(ctx, node_id, wire_types);
    }
    let wire = WireRef { node: node_id, port: Port(0) };
    let ty = wire_types
        .get(&wire)
        .and_then(ConcreteWireType::matrix_type)
        .ok_or_else(|| "GPU static matrix has no validated type".to_owned())?;
    let evaluation = plan_static_matrix(ctx, ty, value, env)?;
    ctx.wire_ids.insert(wire, evaluation);
    Ok(())
}

fn plan_static_matrix(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: &ConcreteMatrixType,
    value: &ConstantMatrix,
    env: &ParamEnv,
) -> Result<PhysicalValueId, String> {
    let mut regular_gadget_digits_per_tower = None;
    if let ConstantMatrix::Gadget { base, small } = value {
        if ty.rows == 0 || !ty.columns.is_multiple_of(ty.rows) {
            return Err("GPU gadget matrix has an invalid shape".into());
        }
        let params = ctx.backend.parameters_on_physical_device(ctx.device, ty)?;
        let base = base
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| error.to_string())?;
        let digit_count = ty.columns / ty.rows;
        let (_, crt_bits, _) = params.to_crt();
        let expected_base = BigInt::from(1u8) << params.base_bits();
        let expected_digits = if *small {
            crt_bits.div_ceil(params.base_bits() as usize)
        } else {
            params.modulus_digits()
        };
        let valid_digits = if *small {
            digit_count == expected_digits
        } else {
            params.gadget_dropped_moduli(Some(digit_count)).is_some()
        };
        if base != expected_base || !valid_digits {
            return Err(format!(
                "GPU gadget layout disagrees with registered parameters: base {base}, digits {digit_count}, expected base {expected_base}, regular digits {expected_digits}"
            ));
        }
        if !small {
            regular_gadget_digits_per_tower = Some(crt_bits.div_ceil(params.base_bits() as usize));
        }
    }
    let canonical = encode_static_matrix(ty, value, env, regular_gadget_digits_per_tower)?;
    let (_coefficient, evaluation, native, _ntt) = allocate_matrix_import_destination(ctx, ty)?;
    // The owner is new at plan time and has no prior GPU or I/O readers.
    unsafe {
        ctx.backend.upload_physical_matrix_import_after_completion(&native, ty, &canonical)?;
    }
    Ok(evaluation)
}

pub(super) fn lower_gadget_trapdoor_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let NodeKind::GadgetTrapdoor { base, .. } = node.kind() else {
        return Err("GPU gadget trapdoor lowerer received another node".into());
    };
    let wire = WireRef { node: node_id, port: Port(0) };
    let trapdoor_ty = wire_types
        .get(&wire)
        .ok_or_else(|| "GPU gadget trapdoor has no validated output".to_owned())?
        .clone();
    let ConcreteWireType::Trapdoor { matrix, .. } = &trapdoor_ty else {
        return Err("GPU gadget trapdoor output is not trapdoor typed".into());
    };
    let public = plan_static_matrix(
        ctx,
        matrix,
        &ConstantMatrix::Gadget { base: base.clone(), small: false },
        env,
    )?;
    let source = ctx
        .values
        .get(public.0 as usize)
        .ok_or_else(|| "GPU gadget public matrix is absent".to_owned())?;
    let mut physical = source.clone();
    physical.ty = trapdoor_ty;
    physical.encodings = Box::new([PhysicalEncoding::PublicGadgetEval]);
    let source_owner =
        ctx.owners.get(&public).ok_or_else(|| "GPU gadget public owner is absent".to_owned())?;
    let resident =
        source_owner.with_physical_view(Arc::new(physical.clone())).map_err(str::to_owned)?;
    let secret = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(secret, Arc::new(resident));
    if let Some(writers) = ctx.producer.get(&public).cloned() {
        ctx.producer.insert(secret, writers);
    }
    ctx.trapdoor_public_ids.insert(secret, public);
    ctx.wire_ids.insert(wire, secret);
    Ok(())
}

/// Lower a validated CRT conversion through its exact coefficient-domain
/// native plan. The destination basis remains ordered as declared by the IR.
pub(super) fn lower_rns_conversion_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &mxx_ir_core::graph::GraphScope,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU RNS conversion arguments are outside their scope".to_owned())?;
    let [input_wire] = arguments.as_slice() else {
        return Err("GPU RNS conversion needs one matrix input".into());
    };
    let source = *ctx
        .wire_ids
        .get(input_wire)
        .ok_or_else(|| "GPU RNS conversion input has no physical value".to_owned())?;
    let source_value = ctx
        .values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU RNS conversion input metadata is absent".to_owned())?;
    let ConcreteWireType::Matrix(source_ty) = &source_value.ty else {
        return Err("GPU RNS conversion input is not an ordinary matrix".into());
    };
    let source_ty = source_ty.clone();
    let source_encodings = source_value.encodings.to_vec();
    let output_wire = WireRef { node: node_id, port: Port(0) };
    let ConcreteWireType::Matrix(destination_ty) = wire_types
        .get(&output_wire)
        .ok_or_else(|| "GPU RNS conversion output has no validated type".to_owned())?
    else {
        return Err("GPU RNS conversion output is not an ordinary matrix".into());
    };
    let destination_ty = destination_ty.clone();
    if source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension() ||
        source_ty.columns != destination_ty.columns
    {
        return Err("GPU RNS conversion dimensions disagree".into());
    }
    let (implementation, parameter) = match node.kind() {
        NodeKind::RnsModUp { digit_size, normalize, .. } => {
            if *digit_size == 0 ||
                !source_ty
                    .ring
                    .crt_moduli()
                    .iter()
                    .all(|prime| destination_ty.ring.crt_moduli().contains(prime)) ||
                source_ty.rows.checked_mul(source_ty.ring.crt_depth().div_ceil(*digit_size)) !=
                    Some(destination_ty.rows)
            {
                return Err("GPU RNS ModUp ordered basis or row geometry disagrees".into());
            }
            let digits = u32::try_from(*digit_size)
                .map_err(|_| "GPU RNS ModUp digit size exceeds u32".to_owned())?;
            (
                GpuImplementation::rns_mod_up(),
                vec![KernelArg::U32(digits), KernelArg::U32(u32::from(*normalize))],
            )
        }
        NodeKind::RnsModDown { plaintext_modulus, .. } => {
            let value = plaintext_modulus
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?;
            let plaintext = value
                .to_biguint()
                .filter(|value| *value >= num_bigint::BigUint::from(2u8))
                .ok_or_else(|| "GPU RNS ModDown plaintext modulus is less than two".to_owned())?
                .to_u64_digits();
            if destination_ty.rows != source_ty.rows ||
                destination_ty.ring.crt_depth() >= source_ty.ring.crt_depth() ||
                !destination_ty
                    .ring
                    .crt_moduli()
                    .iter()
                    .all(|prime| source_ty.ring.crt_moduli().contains(prime))
            {
                return Err("GPU RNS ModDown needs an exact strict ordered-basis subset".into());
            }
            (
                GpuImplementation::rns_mod_down(),
                vec![KernelArg::U64List(plaintext.into_boxed_slice())],
            )
        }
        NodeKind::BlockModSwitch { plaintext_modulus, .. } => {
            let value = plaintext_modulus
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?;
            let words = value
                .to_biguint()
                .filter(|value| *value > num_bigint::BigUint::ZERO)
                .ok_or_else(|| {
                    "GPU block modulus switch plaintext modulus is not positive".to_owned()
                })?
                .to_u64_digits();
            if destination_ty.rows != source_ty.rows ||
                destination_ty.ring.crt_depth() >= source_ty.ring.crt_depth() ||
                !destination_ty
                    .ring
                    .crt_moduli()
                    .iter()
                    .all(|prime| source_ty.ring.crt_moduli().contains(prime))
            {
                return Err(
                    "GPU block modulus switch needs an exact strict ordered-basis subset".into()
                );
            }
            (
                GpuImplementation::block_mod_switch(),
                vec![KernelArg::U64List(words.into_boxed_slice())],
            )
        }
        _ => return Err("GPU RNS conversion lowerer received another node".into()),
    };
    let source_coefficient = match source_encodings.as_slice() {
        [PhysicalEncoding::FullCoeff] => source,
        [PhysicalEncoding::FullEval] => {
            let coefficient =
                allocate_scratch_matrix(ctx, &source_ty, PhysicalEncoding::FullCoeff)?;
            emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], coefficient)?;
            coefficient
        }
        _ => return Err("GPU RNS conversion needs a full matrix input".into()),
    };
    let destination_coefficient =
        allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullCoeff)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source_coefficient)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, destination_coefficient)?;
    let resource_id = u32::try_from(ctx.values.len())
        .ok()
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "GPU RNS resource ID overflows".to_owned())?;
    let implementation = ctx.implementations.register(implementation).map_err(str::to_owned)?;
    let mut arguments = vec![
        KernelArg::U32(resource_id),
        KernelArg::Value(source_coefficient),
        KernelArg::U32(0),
        KernelArg::Value(destination_coefficient),
        KernelArg::U32(0),
    ];
    arguments.extend(parameter);
    arguments.extend([KernelArg::U32(source_binding), KernelArg::U32(destination_binding)]);
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU RNS operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: arguments.into_boxed_slice(),
        outputs: Box::new([destination_coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source_coefficient),
        body: None,
    });
    ctx.producer.insert(
        destination_coefficient,
        vec![(ColumnRange { start: 0, end: destination_ty.columns }, index)],
    );
    let destination = allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(
        ctx,
        GpuImplementation::ntt(false),
        &[destination_coefficient],
        destination,
    )?;
    ctx.wire_ids.insert(output_wire, destination);
    Ok(())
}

/// Recompose rounded coefficient residues in the declared ordered destination
/// CRT basis. Each level updates the same coefficient-domain destination in
/// argument order, so a later level depends on the previous writer.
pub(super) fn lower_crt_recompose_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &mxx_ir_core::graph::GraphScope,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients, .. } = node.kind()
    else {
        return Err("GPU CRT recompose lowerer received another node".into());
    };
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU CRT recompose arguments are outside their scope".to_owned())?;
    if arguments.is_empty() ||
        arguments.len() != plaintext_moduli.len() ||
        arguments.len() != reconstruction_coefficients.len()
    {
        return Err("GPU CRT recompose metadata count differs from inputs".into());
    }
    let output_wire = WireRef { node: node_id, port: Port(0) };
    let ConcreteWireType::Matrix(destination_ty) = wire_types
        .get(&output_wire)
        .ok_or_else(|| "GPU CRT recompose has no validated destination".to_owned())?
    else {
        return Err("GPU CRT recompose destination is not a matrix".into());
    };
    let destination_ty = destination_ty.clone();
    if destination_ty.rows != 1 {
        return Err("GPU CRT recompose destination must have one row".into());
    }
    let destination_coefficient =
        allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullCoeff)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, destination_coefficient)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::crt_recompose_level())
        .map_err(str::to_owned)?;
    for (level, input_wire) in arguments.iter().enumerate() {
        let input = *ctx
            .wire_ids
            .get(input_wire)
            .ok_or_else(|| "GPU CRT recompose input has no physical value".to_owned())?;
        let source_value = &ctx.values[input.0 as usize];
        let ConcreteWireType::Matrix(source_ty) = &source_value.ty else {
            return Err("GPU CRT recompose input is not a matrix".into());
        };
        let source_ty = source_ty.clone();
        let source_encoding = source_value.encodings.to_vec();
        if source_ty.rows != 1 ||
            source_ty.columns != destination_ty.columns ||
            source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension()
        {
            return Err("GPU CRT recompose source geometry differs from destination".into());
        }
        let source_coefficient = match source_encoding.as_slice() {
            [PhysicalEncoding::FullCoeff] => input,
            [PhysicalEncoding::FullEval] => {
                let coefficient =
                    allocate_scratch_matrix(ctx, &source_ty, PhysicalEncoding::FullCoeff)?;
                emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[input], coefficient)?;
                coefficient
            }
            _ => return Err("GPU CRT recompose needs full matrix inputs".into()),
        };
        let plaintext = plaintext_moduli[level]
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| error.to_string())?;
        let source_modulus = source_ty.ring.modulus();
        if plaintext <= BigInt::from(1) || plaintext > source_modulus {
            return Err("GPU CRT recompose plaintext modulus is outside its source ring".into());
        }
        let coefficient = reconstruction_coefficients[level]
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| error.to_string())?;
        if coefficient < BigInt::from(0) || coefficient >= destination_ty.ring.modulus() {
            return Err("GPU CRT recompose coefficient is outside its destination ring".into());
        }
        let plaintext_words = plaintext
            .to_biguint()
            .ok_or_else(|| "GPU CRT recompose plaintext is negative".to_owned())?
            .to_u64_digits();
        let residues = destination_ty
            .ring
            .crt_moduli()
            .iter()
            .map(|modulus| {
                (&coefficient % BigInt::from(*modulus))
                    .to_u64()
                    .ok_or_else(|| "GPU CRT reconstruction residue exceeds u64".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_binding = register_bindings(ctx.bindings, ctx.values, source_coefficient)?;
        let resource_id = *ctx.crt_resource_next;
        *ctx.crt_resource_next = resource_id
            .checked_add(1)
            .ok_or_else(|| "too many GPU CRT recompose resources".to_owned())?;
        let predecessors = all_predecessors(ctx.producer, source_coefficient)
            .into_iter()
            .chain(all_predecessors(ctx.producer, destination_coefficient))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let op_index = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU CRT recompose operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::U32(resource_id),
                KernelArg::Value(source_coefficient),
                KernelArg::U32(0),
                KernelArg::Value(destination_coefficient),
                KernelArg::U32(0),
                KernelArg::U64List(plaintext_words.into_boxed_slice()),
                KernelArg::U64List(residues.into_boxed_slice()),
                KernelArg::U32(u32::from(level == 0)),
                KernelArg::U32(source_binding),
                KernelArg::U32(destination_binding),
            ]),
            outputs: Box::new([destination_coefficient]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors,
            body: None,
        });
        ctx.producer.insert(
            destination_coefficient,
            vec![(ColumnRange { start: 0, end: destination_ty.columns }, op_index)],
        );
    }
    let destination = allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(
        ctx,
        GpuImplementation::ntt(false),
        &[destination_coefficient],
        destination,
    )?;
    ctx.wire_ids.insert(output_wire, destination);
    Ok(())
}

/// Re-encode centered coefficients in an exact destination CRT basis. The
/// source and destination coefficient buffers are separate plan-owned owners.
pub(super) fn lower_centered_rebase_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope: &mxx_ir_core::graph::GraphScope,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    if !matches!(node.kind(), NodeKind::CenteredRebase { .. }) {
        return Err("GPU centered rebase lowerer received another node".into());
    }
    let arguments = scope
        .arguments(node)
        .ok_or_else(|| "GPU centered rebase arguments are outside their scope".to_owned())?;
    let [input_wire] = arguments.as_slice() else {
        return Err("GPU centered rebase needs one matrix input".into());
    };
    let input = *ctx
        .wire_ids
        .get(input_wire)
        .ok_or_else(|| "GPU centered rebase input has no physical value".to_owned())?;
    let source_value = &ctx.values[input.0 as usize];
    let source_ty = source_value
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU centered rebase input has no matrix type".to_owned())?
        .clone();
    let source_type = source_value.ty.clone();
    let source_encoding = source_value.encodings.to_vec();
    let output_wire = WireRef { node: node_id, port: Port(0) };
    let output_type = wire_types
        .get(&output_wire)
        .ok_or_else(|| "GPU centered rebase has no validated output type".to_owned())?
        .clone();
    let destination_ty = output_type
        .matrix_type()
        .ok_or_else(|| "GPU centered rebase output has no matrix type".to_owned())?
        .clone();
    if source_ty.rows != destination_ty.rows ||
        source_ty.columns != destination_ty.columns ||
        source_ty.ring.ring_dimension() != destination_ty.ring.ring_dimension()
    {
        return Err("GPU centered rebase source and destination shapes differ".into());
    }
    let source_coefficient = match (&source_type, source_encoding.as_slice()) {
        (ConcreteWireType::Matrix(_), [PhysicalEncoding::FullCoeff]) => input,
        (ConcreteWireType::Matrix(_), [PhysicalEncoding::FullEval]) => {
            let coefficient =
                allocate_scratch_matrix(ctx, &source_ty, PhysicalEncoding::FullCoeff)?;
            emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[input], coefficient)?;
            coefficient
        }
        (
            ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. },
            [PhysicalEncoding::CompactCoeff { .. }],
        ) => {
            let coefficient =
                allocate_scratch_matrix(ctx, &source_ty, PhysicalEncoding::FullCoeff)?;
            let compact_binding = register_preimage_control_binding(ctx, input)?;
            let coefficient_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
            let implementation = ctx
                .implementations
                .register(GpuImplementation::expand_compact())
                .map_err(str::to_owned)?;
            let index = u32::try_from(ctx.operations.len())
                .map_err(|_| "too many GPU compact expansion operations".to_owned())?;
            ctx.operations.push(CompiledGpuOp {
                implementation,
                arguments: Box::new([
                    KernelArg::Value(input),
                    KernelArg::U32(0),
                    KernelArg::Value(coefficient),
                    KernelArg::U32(0),
                    KernelArg::U32(compact_binding),
                    KernelArg::U32(coefficient_binding),
                ]),
                outputs: Box::new([coefficient]),
                device: ctx.device,
                grid: [1; 3],
                block: [1; 3],
                shared_bytes: 0,
                predecessors: all_predecessors(ctx.producer, input),
                body: None,
            });
            ctx.producer.insert(
                coefficient,
                vec![(ColumnRange { start: 0, end: source_ty.columns }, index)],
            );
            coefficient
        }
        _ => return Err("GPU centered rebase input encoding is unsupported".into()),
    };
    let destination_coefficient =
        allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullCoeff)?;
    emit_crt_convert_coefficients(ctx, source_coefficient, destination_coefficient)?;
    if matches!(output_type, ConcreteWireType::SmallMatrix { .. }) {
        let bound = match &output_type {
            ConcreteWireType::SmallMatrix { max_coefficient_bound, .. } => max_coefficient_bound,
            _ => unreachable!("matched small matrix"),
        };
        let mut bound_words = bound
            .to_biguint()
            .ok_or_else(|| "GPU compact centered rebase bound is negative".to_owned())?
            .to_u64_digits();
        if bound_words.is_empty() {
            bound_words.push(0);
        }
        let (output, _) = allocate_compact_value(ctx, output_type, true)?;
        let control_params = ctx.backend.control_parameters_on_device(ctx.device)?;
        let status = Arc::new(
            GpuExportStatus::new(&control_params, ctx.device).map_err(|error| error.to_string())?,
        );
        let status_id = allocate_preimage_control(
            ctx,
            4,
            BoundStorage::from_export_status(Arc::clone(&status))?,
        )?;
        ctx.control_resets.push(ControlReset::IntegerStatus(status));
        let source_binding = register_bindings(ctx.bindings, ctx.values, destination_coefficient)?;
        let output_binding = register_preimage_control_binding(ctx, output)?;
        let status_binding = register_preimage_control_binding(ctx, status_id)?;
        let implementation = ctx
            .implementations
            .register(GpuImplementation::compact_pack())
            .map_err(str::to_owned)?;
        let index = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU compact pack operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::U32(output.0),
                KernelArg::Value(destination_coefficient),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::Value(status_id),
                KernelArg::U32(0),
                KernelArg::U64List(bound_words.into_boxed_slice()),
                KernelArg::U32(source_binding),
                KernelArg::U32(output_binding),
                KernelArg::U32(status_binding),
            ]),
            outputs: Box::new([output]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors: all_predecessors(ctx.producer, destination_coefficient),
            body: None,
        });
        ctx.producer
            .insert(output, vec![(ColumnRange { start: 0, end: destination_ty.columns }, index)]);
        ctx.wire_ids.insert(output_wire, output);
        return Ok(());
    }
    if !matches!(output_type, ConcreteWireType::Matrix(_)) {
        return Err("GPU centered rebase output has an unsupported semantic kind".into());
    }
    let destination = allocate_scratch_matrix(ctx, &destination_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(
        ctx,
        GpuImplementation::ntt(false),
        &[destination_coefficient],
        destination,
    )?;
    ctx.wire_ids.insert(output_wire, destination);
    Ok(())
}

fn emit_crt_convert_coefficients(
    ctx: &mut PhysicalLoweringContext<'_>,
    source_coefficient: PhysicalValueId,
    destination_coefficient: PhysicalValueId,
) -> Result<(), String> {
    let source_binding = register_bindings(ctx.bindings, ctx.values, source_coefficient)?;
    let destination_binding = register_bindings(ctx.bindings, ctx.values, destination_coefficient)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::crt_convert()).map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU centered rebase operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(source_coefficient),
            KernelArg::U32(0),
            KernelArg::Value(destination_coefficient),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(destination_binding),
        ]),
        outputs: Box::new([destination_coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, source_coefficient),
        body: None,
    });
    let columns = ctx.values[destination_coefficient.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU CRT conversion destination has no matrix type".to_owned())?
        .columns;
    ctx.producer
        .insert(destination_coefficient, vec![(ColumnRange { start: 0, end: columns }, index)]);
    Ok(())
}

pub(super) fn allocate_scratch_matrix(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: &ConcreteMatrixType,
    encoding: PhysicalEncoding,
) -> Result<PhysicalValueId, String> {
    let native = ctx.backend.allocate_physical_matrix(ty, ctx.device, encoding.clone())?;
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU matrix storages".to_owned())?,
    );
    let (physical, owner) = physical_matrix(ty, encoding, storage, native)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, owner);
    Ok(id)
}

fn matrix_view(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    row: usize,
    rows: usize,
    column: usize,
    columns: usize,
) -> Result<PhysicalValueId, String> {
    let source_physical = ctx
        .values
        .get(source.0 as usize)
        .ok_or_else(|| "GPU matrix view source is absent".to_owned())?;
    let source_ty = source_physical
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU matrix view source is not a matrix".to_owned())?;
    if rows == 0 ||
        columns == 0 ||
        row.checked_add(rows).is_none_or(|end| end > source_ty.rows) ||
        column.checked_add(columns).is_none_or(|end| end > source_ty.columns)
    {
        return Err("GPU matrix view lies outside its allocation".into());
    }
    let mut view = source_physical.clone();
    // The view carries a cropped window into the owner's full semantic
    // matrix. Both the semantic origin and the owner-derived byte address
    // must move; native raw kernels address rows and columns from that base.
    let row = u64::try_from(row).map_err(|_| "GPU matrix row offset exceeds u64")?;
    let rows = u64::try_from(rows).map_err(|_| "GPU matrix row count exceeds u64")?;
    let column = u64::try_from(column).map_err(|_| "GPU matrix column offset exceeds u64")?;
    let columns = u64::try_from(columns).map_err(|_| "GPU matrix column count exceeds u64")?;
    for part in view.parts.iter_mut() {
        if row.checked_add(rows).is_none_or(|end| end > part.view.extent[0]) ||
            column.checked_add(columns).is_none_or(|end| end > part.view.extent[1])
        {
            return Err("GPU matrix view exceeds its source window".into());
        }
        let row_bytes =
            row.checked_mul(part.view.byte_strides[0]).ok_or("GPU matrix row address overflows")?;
        let column_bytes = column
            .checked_mul(part.view.byte_strides[1])
            .ok_or("GPU matrix column address overflows")?;
        part.view.byte_offset = part
            .view
            .byte_offset
            .checked_add(row_bytes)
            .and_then(|offset| offset.checked_add(column_bytes))
            .ok_or("GPU matrix view byte offset overflows")?;
        part.view.origin[0] =
            part.view.origin[0].checked_add(row).ok_or("GPU matrix row origin overflows")?;
        part.view.origin[1] =
            part.view.origin[1].checked_add(column).ok_or("GPU matrix column origin overflows")?;
        part.view.extent[0] = rows;
        part.view.extent[1] = columns;
    }
    let source_owner =
        ctx.owners.get(&source).ok_or_else(|| "GPU matrix view owner is absent".to_owned())?;
    let owner =
        Arc::new(source_owner.with_physical_view(Arc::new(view.clone())).map_err(str::to_owned)?);
    let id = value_id(ctx.values.len())?;
    ctx.values.push(view);
    ctx.owners.insert(id, owner);
    Ok(id)
}

/// A compact publication addresses the full owner base and uses the semantic
/// column origin to scatter a packed tile into that owner. Unlike a full
/// matrix subview, its byte offset must not advance by the column stride.
fn compact_column_view(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    column: usize,
    columns: usize,
) -> Result<PhysicalValueId, String> {
    let physical = ctx.values.get(source.0 as usize).ok_or("GPU compact source is absent")?;
    let matrix = physical.ty.matrix_type().ok_or("GPU compact source has no matrix type")?;
    if columns == 0 || column.checked_add(columns).is_none_or(|end| end > matrix.columns) {
        return Err("GPU compact tile lies outside its full owner".into());
    }
    if !matches!(physical.encodings.as_ref(), [PhysicalEncoding::CompactCoeff { .. }]) ||
        physical.parts.len() != 1
    {
        return Err("GPU preimage tile needs one global compact owner".into());
    }
    let mut view = physical.clone();
    let part = &mut view.parts[0];
    part.view.origin[1] = u64::try_from(column).map_err(|_| "GPU tile column exceeds u64")?;
    part.view.extent[1] = u64::try_from(columns).map_err(|_| "GPU tile width exceeds u64")?;
    let owner = ctx.owners.get(&source).ok_or("GPU compact owner is absent")?;
    let resident = owner.with_physical_view(Arc::new(view.clone())).map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(view);
    ctx.owners.insert(id, Arc::new(resident));
    Ok(id)
}

/// `id` in full-Eval encoding. A full-Coeff matrix gets one forward NTT that
/// `ctx.converted` shares with every later consumer; other values pass through.
pub(super) fn full_eval_value(
    ctx: &mut PhysicalLoweringContext<'_>,
    id: PhysicalValueId,
) -> Result<PhysicalValueId, String> {
    let physical = &ctx.values[id.0 as usize];
    let (Some(ty), [PhysicalEncoding::FullCoeff]) =
        (physical.ty.matrix_type(), physical.encodings.as_ref())
    else {
        return Ok(id);
    };
    if let Some(&converted) = ctx.converted.get(&id) {
        return Ok(converted);
    }
    let ty = ty.clone();
    let evaluation = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[id], evaluation)?;
    ctx.converted.insert(id, evaluation);
    ctx.converted.insert(evaluation, id);
    Ok(evaluation)
}

pub(super) fn emit_matrix_operation(
    ctx: &mut PhysicalLoweringContext<'_>,
    implementation: GpuImplementation,
    inputs: &[PhysicalValueId],
    output: PhysicalValueId,
) -> Result<u32, String> {
    let implementation =
        ctx.implementations.register(implementation.clone()).map_err(str::to_owned)?;
    let output_binding = register_bindings(ctx.bindings, ctx.values, output)?;
    let arguments = match inputs {
        [source] => {
            let source_binding = register_bindings(ctx.bindings, ctx.values, *source)?;
            Box::new([
                KernelArg::Value(*source),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::U32(source_binding),
                KernelArg::U32(output_binding),
            ]) as Box<[KernelArg]>
        }
        [left, right] => {
            let left_binding = register_bindings(ctx.bindings, ctx.values, *left)?;
            let right_binding = register_bindings(ctx.bindings, ctx.values, *right)?;
            Box::new([
                KernelArg::Value(*left),
                KernelArg::U32(0),
                KernelArg::Value(*right),
                KernelArg::U32(0),
                KernelArg::Value(output),
                KernelArg::U32(0),
                KernelArg::U32(left_binding),
                KernelArg::U32(right_binding),
                KernelArg::U32(output_binding),
            ])
        }
        _ => return Err("GPU matrix operation needs one or two inputs".into()),
    };
    let predecessors = inputs
        .iter()
        .flat_map(|id| all_predecessors(ctx.producer, *id))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU matrix operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments,
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    let width = ctx.values[output.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU matrix operation output has no matrix type".to_owned())?
        .columns;
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: width }, index)]);
    Ok(index)
}

fn allocate_device_seed(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    label: u8,
) -> Result<PhysicalValueId, String> {
    let site = mxx_ir_core::encoding::hash_canonical(&(scope_id, node_id, label, ctx.values.len()))
        .map_err(|error| error.to_string())?;
    let params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let seed_native =
        Arc::new(GpuDeviceSeed::new(&params, ctx.device).map_err(|error| error.to_string())?);
    seed_native.upload(&[0; 32]).map_err(|error| error.to_string())?;
    let seed_storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU sample seeds".to_owned())?,
    );
    let seed_physical = PhysicalValue {
        ty: ConcreteWireType::Bytes { length: 32 },
        encodings: Box::new([PhysicalEncoding::Bytes]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage: seed_storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([32]),
                byte_strides: Box::new([1]),
                element_bytes: 1,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let seed_resident = GpuResidentValue::new(
        Arc::new(seed_physical.clone()),
        BTreeMap::from([(seed_storage, BoundStorage::from_device_seed(Arc::clone(&seed_native))?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let seed = value_id(ctx.values.len())?;
    ctx.values.push(seed_physical);
    ctx.owners.insert(seed, Arc::new(seed_resident));
    ctx.sample_seeds.push(SampleSeed { owner: seed_native, site });
    Ok(seed)
}

fn allocate_preimage_control(
    ctx: &mut PhysicalLoweringContext<'_>,
    bytes: usize,
    bound: BoundStorage,
) -> Result<PhysicalValueId, String> {
    let storage = StorageRef::Scratch(
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU preimage controls".to_owned())?,
    );
    if bound.device != ctx.device || bound.bytes != bytes as u64 {
        return Err("GPU preimage control owner has the wrong device or width".into());
    }
    let physical = PhysicalValue {
        ty: ConcreteWireType::Bytes { length: bytes },
        encodings: Box::new([PhysicalEncoding::Bytes]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device: ctx.device,
            view: PhysicalView {
                byte_offset: 0,
                origin: Box::new([0]),
                extent: Box::new([bytes as u64]),
                byte_strides: Box::new([1]),
                element_bytes: 1,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, bound)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(resident));
    Ok(id)
}

fn compact_value_owner(
    backend: &GpuDcrtBackend,
    device: i32,
    ty: ConcreteWireType,
    storage: StorageRef,
) -> Result<(PhysicalValue, Arc<GpuResidentValue>, Arc<GpuSmallMatrix>, usize), String> {
    let (matrix, max_coefficient_bound, bound_domain) = match &ty {
        ConcreteWireType::Preimage { matrix, max_coefficient_bound, bound_domain } |
        ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound, bound_domain } => {
            (matrix, max_coefficient_bound, bound_domain)
        }
        _ => return Err("GPU compact destination has no bounded matrix type".into()),
    };
    let bound = max_coefficient_bound
        .to_biguint()
        .ok_or_else(|| "GPU preimage bound is negative".to_owned())?;
    let params = backend.parameters_on_physical_device(device, matrix)?;
    let descriptor = GpuSmallMatrixOutputDescriptor::for_shape_in_domain(
        params,
        matrix.rows,
        matrix.columns,
        bound,
        *bound_domain,
    )
    .map_err(|error| error.to_string())?;
    let owner = Arc::new(descriptor.allocate().map_err(|error| error.to_string())?);
    owner.wait_until_ready();
    let binding = owner.binding_descriptor().map_err(|error| error.to_string())?;
    if binding.physical_device != device ||
        binding.rows != matrix.rows ||
        binding.columns != matrix.columns ||
        binding.ring_dimension != matrix.ring.ring_dimension() as usize ||
        binding.crt_depth !=
            if *bound_domain == CoefficientBoundDomain::PerCrtLimb {
                matrix.ring.crt_depth()
            } else {
                1
            } ||
        binding.bound_domain != *bound_domain ||
        binding.magnitude_bytes != descriptor.magnitude_bytes ||
        binding.payload_bytes != descriptor.payload_bytes
    {
        return Err("GPU compact output descriptor disagrees with its owner".into());
    }
    let width = descriptor.magnitude_bytes + 1;
    let per_limb = *bound_domain == CoefficientBoundDomain::PerCrtLimb;
    let (encoding, extent, byte_strides) = if per_limb {
        (
            PhysicalEncoding::CompactCoeffPerCrtLimb {
                magnitude_bytes: descriptor.magnitude_bytes,
            },
            vec![
                matrix.rows as u64,
                matrix.columns as u64,
                matrix.ring.ring_dimension() as u64,
                matrix.ring.crt_depth() as u64,
                width as u64,
            ],
            vec![
                binding.row_stride_bytes as u64,
                binding.column_stride_bytes as u64,
                binding.coefficient_stride_bytes as u64,
                binding.limb_stride_bytes as u64,
                1,
            ],
        )
    } else {
        (
            PhysicalEncoding::CompactCoeff { magnitude_bytes: descriptor.magnitude_bytes },
            vec![
                matrix.rows as u64,
                matrix.columns as u64,
                matrix.ring.ring_dimension() as u64,
                width as u64,
            ],
            vec![
                binding.row_stride_bytes as u64,
                binding.column_stride_bytes as u64,
                binding.coefficient_stride_bytes as u64,
                1,
            ],
        )
    };
    let physical = PhysicalValue {
        ty: ty.clone(),
        encodings: Box::new([encoding]),
        parts: Box::new([PhysicalPart {
            leaf: 0,
            storage,
            device,
            view: PhysicalView {
                byte_offset: 0,
                origin: vec![0; extent.len()].into_boxed_slice(),
                extent: extent.into_boxed_slice(),
                byte_strides: byte_strides.into_boxed_slice(),
                element_bytes: 1,
            },
        }]),
        integer_ranges: BTreeMap::new(),
    };
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, BoundStorage::from_small_matrix_payload(Arc::clone(&owner))?)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    Ok((physical, Arc::new(resident), owner, descriptor.magnitude_bytes))
}

pub(super) fn allocate_compact_value(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    returned: bool,
) -> Result<(PhysicalValueId, usize), String> {
    let storage_number =
        u32::try_from(ctx.values.len()).map_err(|_| "too many GPU compact storages".to_owned())?;
    let storage = if returned {
        StorageRef::Output(storage_number)
    } else {
        StorageRef::Scratch(storage_number)
    };
    let (physical, resident, _, magnitude_bytes) =
        compact_value_owner(ctx.backend, ctx.device, ty, storage)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, resident);
    Ok((id, magnitude_bytes))
}

fn emit_fresh_matrix_sample_parts(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    label: u8,
    ty: &ConcreteMatrixType,
    gaussian_sigma: Option<f64>,
    seed_override: Option<PhysicalValueId>,
) -> Result<(PhysicalValueId, PhysicalValueId), String> {
    let seed = match seed_override {
        Some(seed) => seed,
        None => allocate_device_seed(ctx, scope_id, node_id, label)?,
    };
    let coefficient = allocate_scratch_matrix(ctx, ty, PhysicalEncoding::FullCoeff)?;
    let evaluation = allocate_scratch_matrix(ctx, ty, PhysicalEncoding::FullEval)?;
    let seed_binding =
        u32::try_from(ctx.bindings.len()).map_err(|_| "too many GPU sample bindings".to_owned())?;
    ctx.bindings.push(GpuBindingSource::PhysicalPart { value: seed, part: 0, limb: 0 });
    let coefficient_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
    let sample = ctx
        .implementations
        .register(GpuImplementation::sample(gaussian_sigma.is_some()))
        .map_err(str::to_owned)?;
    let sample_index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU sample operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation: sample,
        arguments: Box::new([
            KernelArg::Value(coefficient),
            KernelArg::U32(0),
            KernelArg::Value(seed),
            KernelArg::U32(0),
            KernelArg::F64(gaussian_sigma.unwrap_or(0.0)),
            KernelArg::U64(if gaussian_sigma.is_some() { u64::MAX } else { 0 }),
            KernelArg::U64(if gaussian_sigma.is_some() { 0 } else { 1 }),
            KernelArg::I64(0),
            KernelArg::I64(0),
            KernelArg::U64(ty.columns as u64),
            KernelArg::U64(0),
            KernelArg::U32(coefficient_binding),
            KernelArg::U32(seed_binding),
        ]),
        outputs: Box::new([coefficient]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: Box::new([]),
        body: None,
    });
    ctx.producer
        .insert(coefficient, vec![(ColumnRange { start: 0, end: ty.columns }, sample_index)]);
    emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[coefficient], evaluation)?;
    Ok((coefficient, evaluation))
}

fn emit_fresh_matrix_sample(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    label: u8,
    ty: &ConcreteMatrixType,
    gaussian_sigma: Option<f64>,
) -> Result<PhysicalValueId, String> {
    emit_fresh_matrix_sample_parts(ctx, scope_id, node_id, label, ty, gaussian_sigma, None)
        .map(|(_, evaluation)| evaluation)
}

fn emit_matrix_fill(
    ctx: &mut PhysicalLoweringContext<'_>,
    output: PhysicalValueId,
    implementation: GpuImplementation,
    parameters: Box<[KernelArg]>,
) -> Result<u32, String> {
    let implementation = ctx.implementations.register(implementation).map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU matrix fill operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: parameters,
        outputs: Box::new([output]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: Box::new([]),
        body: None,
    });
    let width = ctx.values[output.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU fill output has no matrix type".to_owned())?
        .columns;
    ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: width }, index)]);
    Ok(index)
}

pub(super) fn pack_trapdoor_leaves(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    leaves: [PhysicalValueId; 6],
    encoding: PhysicalEncoding,
) -> Result<PhysicalValueId, String> {
    let mut parts = Vec::new();
    let mut storage = BTreeMap::new();
    let mut predecessors = BTreeSet::new();
    for (leaf, source) in leaves.into_iter().enumerate() {
        let physical = ctx
            .values
            .get(source.0 as usize)
            .ok_or_else(|| "GPU trapdoor leaf is absent".to_owned())?;
        if physical.encodings.as_ref() != [encoding.clone()] {
            return Err("GPU trapdoor leaves have mixed physical encodings".into());
        }
        let owner = ctx
            .owners
            .get(&source)
            .ok_or_else(|| "GPU trapdoor leaf owner is absent".to_owned())?;
        for part in physical.parts.iter() {
            let mut part = part.clone();
            part.leaf = leaf as u32;
            storage.insert(
                part.storage,
                owner
                    .storage(part.storage)
                    .ok_or_else(|| "GPU trapdoor leaf storage is absent".to_owned())?
                    .clone(),
            );
            parts.push(part);
        }
        predecessors.extend(all_predecessors(ctx.producer, source));
    }
    let physical = PhysicalValue {
        ty,
        encodings: vec![encoding; 6].into_boxed_slice(),
        parts: parts.into_boxed_slice(),
        integer_ranges: BTreeMap::new(),
    };
    let owner = GpuResidentValue::new(Arc::new(physical.clone()), storage, Box::new([]))
        .map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(owner));
    ctx.producer.insert(
        id,
        predecessors.into_iter().map(|op| (ColumnRange { start: 0, end: 1 }, op)).collect(),
    );
    Ok(id)
}

pub(super) fn trapdoor_leaf_types(
    ty: &ConcreteWireType,
) -> Result<[ConcreteMatrixType; 6], String> {
    let ConcreteWireType::Trapdoor { matrix, digit_count, .. } = ty else {
        return Err("GPU trapdoor leaf layout requires a trapdoor type".into());
    };
    let d = matrix.rows;
    let dk = d
        .checked_mul(*digit_count)
        .ok_or_else(|| "GPU trapdoor leaf width overflows".to_owned())?;
    let shape = |rows, columns| ConcreteMatrixType { ring: matrix.ring.clone(), rows, columns };
    Ok([shape(d, dk), shape(d, dk), shape(d, d), shape(d, d), shape(d, d), shape(2 * d, dk)])
}

fn register_all_parts(
    bindings: &mut Vec<GpuBindingSource>,
    values: &[PhysicalValue],
    id: PhysicalValueId,
) -> Result<u32, String> {
    let value =
        values.get(id.0 as usize).ok_or_else(|| "GPU composite value is absent".to_owned())?;
    let base =
        u32::try_from(bindings.len()).map_err(|_| "too many GPU composite bindings".to_owned())?;
    for part in 0..value.parts.len() {
        bindings.push(GpuBindingSource::PhysicalPart {
            value: id,
            part: u32::try_from(part)
                .map_err(|_| "GPU composite part index exceeds u32".to_owned())?,
            limb: 0,
        });
    }
    Ok(base)
}

fn emit_trapdoor_leaf_copies(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    destination: PhysicalValueId,
) -> Result<(), String> {
    let source_ty = ctx.values[source.0 as usize].ty.clone();
    if source_ty != ctx.values[destination.0 as usize].ty {
        return Err("GPU trapdoor return copy changes semantic type".into());
    }
    let depth = source_ty
        .matrix_type()
        .ok_or_else(|| "GPU trapdoor has no matrix type".to_owned())?
        .ring
        .crt_depth();
    if ctx.values[source.0 as usize].parts.len() != 6 * depth ||
        ctx.values[destination.0 as usize].parts.len() != 6 * depth
    {
        return Err("GPU trapdoor copy does not cover six ordered CRT leaves".into());
    }
    let source_base = register_all_parts(ctx.bindings, ctx.values, source)?;
    let destination_base = register_all_parts(ctx.bindings, ctx.values, destination)?;
    let implementation = ctx
        .implementations
        .register(GpuImplementation::matrix_copy_view())
        .map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, source);
    let mut writers = Vec::new();
    for leaf in 0..6 {
        let part = u32::try_from(leaf * depth)
            .map_err(|_| "GPU trapdoor copy part index exceeds u32".to_owned())?;
        let index = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU trapdoor copy operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(source),
                KernelArg::U32(part),
                KernelArg::Value(destination),
                KernelArg::U32(part),
                KernelArg::U32(source_base + part),
                KernelArg::U32(destination_base + part),
            ]),
            outputs: Box::new([destination]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors: predecessors.clone(),
            body: None,
        });
        writers.push((ColumnRange { start: 0, end: 1 }, index));
    }
    ctx.producer.insert(destination, writers);
    Ok(())
}

fn inverse_trapdoor_for_export(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
) -> Result<PhysicalValueId, String> {
    let ty = ctx.values[source.0 as usize].ty.clone();
    let leaf_types = trapdoor_leaf_types(&ty)?;
    let depth = leaf_types[0].ring.crt_depth();
    let source_base = register_all_parts(ctx.bindings, ctx.values, source)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::ntt(true)).map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, source);
    let mut leaves = Vec::new();
    for (leaf, leaf_ty) in leaf_types.iter().enumerate() {
        let coefficient = allocate_scratch_matrix(ctx, leaf_ty, PhysicalEncoding::FullCoeff)?;
        let destination_binding = register_bindings(ctx.bindings, ctx.values, coefficient)?;
        let source_part = u32::try_from(leaf * depth)
            .map_err(|_| "GPU trapdoor iNTT part exceeds u32".to_owned())?;
        let index = u32::try_from(ctx.operations.len())
            .map_err(|_| "too many GPU trapdoor iNTT operations".to_owned())?;
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(source),
                KernelArg::U32(source_part),
                KernelArg::Value(coefficient),
                KernelArg::U32(0),
                KernelArg::U32(source_base + source_part),
                KernelArg::U32(destination_binding),
            ]),
            outputs: Box::new([coefficient]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors: predecessors.clone(),
            body: None,
        });
        ctx.producer
            .insert(coefficient, vec![(ColumnRange { start: 0, end: leaf_ty.columns }, index)]);
        leaves.push(coefficient);
    }
    let leaves: [PhysicalValueId; 6] =
        leaves.try_into().map_err(|_| "GPU trapdoor iNTT has wrong leaf count".to_owned())?;
    pack_trapdoor_leaves(ctx, ty, leaves, PhysicalEncoding::FullCoeff)
}

fn inverse_matrix_for_export(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
) -> Result<PhysicalValueId, String> {
    let ty = ctx.values[source.0 as usize]
        .ty
        .matrix_type()
        .ok_or_else(|| "GPU artifact output is not a matrix".to_owned())?
        .clone();
    if ctx.values[source.0 as usize].encodings.as_ref() != [PhysicalEncoding::FullEval] {
        return Err("GPU matrix artifact source is not full Eval".into());
    }
    let coefficient = allocate_scratch_matrix(ctx, &ty, PhysicalEncoding::FullCoeff)?;
    emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[source], coefficient)?;
    Ok(coefficient)
}

fn trapdoor_leaf_view(
    ctx: &mut PhysicalLoweringContext<'_>,
    secret: PhysicalValueId,
    leaf: usize,
) -> Result<PhysicalValueId, String> {
    let secret_physical = ctx
        .values
        .get(secret.0 as usize)
        .ok_or_else(|| "GPU trapdoor leaf source is missing".to_owned())?;
    let leaf_ty = trapdoor_leaf_types(&secret_physical.ty)?
        .get(leaf)
        .ok_or_else(|| "GPU trapdoor leaf index is invalid".to_owned())?
        .clone();
    let encoding = secret_physical
        .encodings
        .get(leaf)
        .ok_or_else(|| "GPU trapdoor leaf encoding is missing".to_owned())?
        .clone();
    let parts = secret_physical
        .parts
        .iter()
        .filter(|part| part.leaf as usize == leaf)
        .map(|part| {
            let mut selected = part.clone();
            selected.leaf = 0;
            selected
        })
        .collect::<Vec<_>>();
    if parts.len() != leaf_ty.ring.crt_depth() {
        return Err("GPU trapdoor leaf has incomplete ordered CRT parts".into());
    }
    let physical = PhysicalValue {
        ty: ConcreteWireType::Matrix(leaf_ty),
        encodings: Box::new([encoding]),
        parts: parts.into_boxed_slice(),
        integer_ranges: BTreeMap::new(),
    };
    let source =
        ctx.owners.get(&secret).ok_or_else(|| "GPU trapdoor leaf owner is missing".to_owned())?;
    let resident = source.with_physical_view(Arc::new(physical.clone())).map_err(str::to_owned)?;
    let id = value_id(ctx.values.len())?;
    ctx.values.push(physical);
    ctx.owners.insert(id, Arc::new(resident));
    if let Some(writers) = ctx.producer.get(&secret).cloned() {
        ctx.producer.insert(id, writers);
    }
    Ok(id)
}

fn push_preimage_op(
    ctx: &mut PhysicalLoweringContext<'_>,
    implementation: GpuImplementation,
    arguments: Box<[KernelArg]>,
    outputs: Box<[PhysicalValueId]>,
    predecessors: Box<[u32]>,
) -> Result<u32, String> {
    let implementation = ctx.implementations.register(implementation).map_err(str::to_owned)?;
    let index = u32::try_from(ctx.operations.len())
        .map_err(|_| "too many GPU preimage operations".to_owned())?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments,
        outputs,
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors,
        body: None,
    });
    Ok(index)
}

fn register_preimage_workspace(
    bindings: &mut Vec<GpuBindingSource>,
    kind: GpuPreparedWorkspaceKind,
    resource_id: u32,
    components: u32,
) -> Result<u32, String> {
    let first = u32::try_from(bindings.len())
        .map_err(|_| "too many GPU preimage workspace bindings".to_owned())?;
    for component in 0..components {
        bindings.push(GpuBindingSource::PreparedWorkspace { kind, resource_id, component });
    }
    Ok(first)
}

pub(super) fn register_preimage_control_binding(
    ctx: &mut PhysicalLoweringContext<'_>,
    value: PhysicalValueId,
) -> Result<u32, String> {
    if ctx.values.get(value.0 as usize).is_none_or(|physical| physical.parts.len() != 1) {
        return Err("GPU preimage control requires one physical part".into());
    }
    let binding = u32::try_from(ctx.bindings.len())
        .map_err(|_| "too many GPU control bindings".to_owned())?;
    ctx.bindings.push(GpuBindingSource::PhysicalPart { value, part: 0, limb: 0 });
    Ok(binding)
}

fn derive_preimage_stage_seed(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    attempt: PhysicalValueId,
    domain: u64,
) -> Result<PhysicalValueId, String> {
    let label = u8::try_from(domain).map_err(|_| "GPU preimage seed domain exceeds u8")?;
    let base = allocate_device_seed(ctx, scope_id, node_id, 10 + 2 * label)?;
    let derived = allocate_device_seed(ctx, scope_id, node_id, 11 + 2 * label)?;
    let base_binding = register_preimage_control_binding(ctx, base)?;
    let attempt_binding = register_preimage_control_binding(ctx, attempt)?;
    let derived_binding = register_preimage_control_binding(ctx, derived)?;
    let index = push_preimage_op(
        ctx,
        GpuImplementation::preimage_derive_attempt_seed(),
        Box::new([
            KernelArg::Value(base),
            KernelArg::U32(0),
            KernelArg::Value(attempt),
            KernelArg::U32(0),
            KernelArg::U64(domain),
            KernelArg::Value(derived),
            KernelArg::U32(0),
            KernelArg::U32(base_binding),
            KernelArg::U32(attempt_binding),
            KernelArg::U32(derived_binding),
        ]),
        Box::new([derived]),
        Box::new([]),
    )?;
    ctx.producer.insert(derived, vec![(ColumnRange { start: 0, end: 1 }, index)]);
    Ok(derived)
}

/// Register a hash sampler's tag as a plan resource: the static prefix and
/// constant components, plus each resident integer operand and its binding.
pub(super) fn hash_tag_resource(
    ctx: &mut PhysicalLoweringContext<'_>,
    arguments: &[WireRef],
    tag_prefix: &[u8],
    tag_components: &[HashTagComponent],
    env: &ParamEnv,
) -> Result<(u32, Vec<(PhysicalValueId, u32, u32)>), String> {
    let mut parts = vec![GpuHashTagPart::prefix(tag_prefix)];
    let mut operands = Vec::new();
    for component in tag_components {
        let part = match component {
            HashTagComponent::Bytes(bytes) => {
                GpuHashTagPart::bytes_component(bytes).map_err(|error| error.to_string())?
            }
            HashTagComponent::Integer(expression) => {
                let value = expression
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| error.to_string())?;
                GpuHashTagPart::integer_constant(&value).map_err(|error| error.to_string())?
            }
            HashTagComponent::Decimal(expression) => {
                let value = expression
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| error.to_string())?;
                GpuHashTagPart::decimal_constant(&value).map_err(|error| error.to_string())?
            }
            HashTagComponent::U64Le(expression) => {
                let value = expression
                    .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                    .map_err(|error| error.to_string())?
                    .to_u64()
                    .ok_or("GPU hash U64Le component exceeds u64")?;
                GpuHashTagPart::u64_le_constant(value)
            }
            HashTagComponent::Operand(index) => {
                let wire = arguments
                    .get(*index)
                    .ok_or("GPU hash operand index is outside its arguments")?;
                let id = *ctx.wire_ids.get(wire).ok_or("GPU hash operand is not physical")?;
                let physical = &ctx.values[id.0 as usize];
                if !matches!(physical.ty, ConcreteWireType::Int | ConcreteWireType::ConstantInt) ||
                    !matches!(physical.encodings.as_ref(), [PhysicalEncoding::Signed(_)]) ||
                    physical.parts.len() != 1
                {
                    return Err("GPU hash operand must be a resident signed scalar".into());
                }
                let binding = register_preimage_control_binding(ctx, id)?;
                let operand_index = operands.len();
                operands.push((id, 0, binding));
                GpuHashTagPart::Integer(operand_index)
            }
        };
        parts.push(part);
    }
    let resource_id = *ctx.crt_resource_next;
    *ctx.crt_resource_next = resource_id.checked_add(1).ok_or("GPU hash resource ID overflows")?;
    if ctx
        .hash_resources
        .insert(
            resource_id,
            GpuHashResourceSpec {
                parts: Arc::from(parts),
                operands: operands.clone().into_boxed_slice(),
            },
        )
        .is_some()
    {
        return Err("GPU hash resource ID is duplicated".into());
    }
    Ok((resource_id, operands))
}

/// Emit one hash sample of `resource_id` keyed by `key` into `destination`,
/// a full-Coeff matrix or an integer family, reporting into a fresh status.
pub(super) fn push_hash_sample(
    ctx: &mut PhysicalLoweringContext<'_>,
    resource_id: u32,
    key: PhysicalValueId,
    destination: PhysicalValueId,
    destination_binding: u32,
    operands: &[(PhysicalValueId, u32, u32)],
) -> Result<u32, String> {
    let control_params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let status_owner = Arc::new(
        GpuExportStatus::new(&control_params, ctx.device).map_err(|error| error.to_string())?,
    );
    let status = allocate_preimage_control(
        ctx,
        4,
        BoundStorage::from_export_status(Arc::clone(&status_owner))?,
    )?;
    ctx.control_resets.push(ControlReset::IntegerStatus(status_owner));
    let key_binding = register_preimage_control_binding(ctx, key)?;
    let status_binding = register_preimage_control_binding(ctx, status)?;
    let mut predecessors = all_predecessors(ctx.producer, key).into_vec();
    for (id, _, _) in operands {
        predecessors.extend(all_predecessors(ctx.producer, *id));
    }
    predecessors.sort_unstable();
    predecessors.dedup();
    push_preimage_op(
        ctx,
        GpuImplementation::hash_sample(),
        Box::new([
            KernelArg::U32(resource_id),
            KernelArg::Value(key),
            KernelArg::U32(0),
            KernelArg::Value(destination),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U32(key_binding),
            KernelArg::U32(destination_binding),
            KernelArg::U32(status_binding),
        ]),
        Box::new([destination]),
        predecessors.into_boxed_slice(),
    )
}

pub(super) fn lower_hash_sample_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let NodeKind::HashSample { variant, tag_prefix, tag_components, base, digit_count, .. } =
        node.kind()
    else {
        return Err("GPU hash lowerer received another node".into());
    };
    let scope = ctx
        .validated
        .source
        .scope(scope_id)
        .ok_or_else(|| "GPU hash scope is missing".to_owned())?;
    let arguments = scope.arguments(node).ok_or("GPU hash arguments are missing")?;
    let (key_wire, _) = arguments.split_first().ok_or("GPU hash key is missing")?;
    let key = *ctx.wire_ids.get(key_wire).ok_or("GPU hash key has no physical value")?;
    let key_physical = &ctx.values[key.0 as usize];
    if key_physical.ty != (ConcreteWireType::Bytes { length: 32 }) ||
        key_physical.encodings.as_ref() != [PhysicalEncoding::Bytes] ||
        key_physical.parts.len() != 1
    {
        return Err("GPU hash key must be resident Bytes32".into());
    }
    let output_wire = WireRef { node: node_id, port: Port(0) };
    let output_ty =
        wire_types.get(&output_wire).ok_or("GPU hash output has no validated type")?.clone();
    let output_matrix =
        output_ty.matrix_type().ok_or("GPU hash output has no matrix type")?.clone();
    let (source_rows, decomposition) = match variant {
        HashVariant::Plain if base.is_none() && digit_count.is_none() => {
            if !matches!(output_ty, ConcreteWireType::Matrix(_)) {
                return Err("GPU plain hash output is not a full matrix".into());
            }
            (output_matrix.rows, None)
        }
        HashVariant::Decomposed | HashVariant::SmallDecomposed => {
            let base = base
                .as_ref()
                .ok_or("GPU decomposed hash has no gadget base")?
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?;
            let count = digit_count
                .as_ref()
                .ok_or("GPU decomposed hash has no digit count")?
                .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
                .map_err(|error| error.to_string())?
                .to_usize()
                .filter(|count| *count > 0 && output_matrix.rows.is_multiple_of(*count))
                .ok_or("GPU hash digit count is invalid")?;
            let params = ctx.backend.parameters_on_physical_device(ctx.device, &output_matrix)?;
            if base != (BigInt::from(1u8) << params.base_bits()) ||
                !matches!(output_ty, ConcreteWireType::SmallMatrix { .. })
            {
                return Err("GPU hash decomposition layout differs from its backend".into());
            }
            let dropped = if matches!(variant, HashVariant::SmallDecomposed) {
                if count != params.crt_bits().div_ceil(params.base_bits() as usize) {
                    return Err("GPU small hash digit count differs from its backend".into());
                }
                if !matches!(
                    output_ty,
                    ConcreteWireType::SmallMatrix {
                        bound_domain: CoefficientBoundDomain::PerCrtLimb,
                        ..
                    }
                ) {
                    return Err("GPU small hash needs per-CRT-limb bounded output".into());
                }
                None
            } else {
                if !matches!(
                    output_ty,
                    ConcreteWireType::SmallMatrix {
                        bound_domain: CoefficientBoundDomain::Global,
                        ..
                    }
                ) {
                    return Err("GPU regular hash needs globally bounded output".into());
                }
                Some(
                    params
                        .gadget_dropped_moduli(Some(count))
                        .ok_or("GPU hash decomposition digit count is unsupported")?,
                )
            };
            (output_matrix.rows / count, Some(dropped))
        }
        HashVariant::Plain => return Err("GPU plain hash has decomposition metadata".into()),
    };
    let source_ty = ConcreteMatrixType {
        ring: output_matrix.ring.clone(),
        rows: source_rows,
        columns: output_matrix.columns,
    };
    let (resource_id, operands) =
        hash_tag_resource(ctx, &arguments, tag_prefix, tag_components, env)?;
    let sampled = allocate_scratch_matrix(ctx, &source_ty, PhysicalEncoding::FullCoeff)?;
    let sampled_binding = register_bindings(ctx.bindings, ctx.values, sampled)?;
    let sample = push_hash_sample(ctx, resource_id, key, sampled, sampled_binding, &operands)?;
    let control_params = ctx.backend.control_parameters_on_device(ctx.device)?;
    ctx.producer.insert(sampled, vec![(ColumnRange { start: 0, end: source_ty.columns }, sample)]);
    if let Some(dropped) = decomposition {
        let decomposed = allocate_scratch_matrix(ctx, &output_matrix, PhysicalEncoding::FullCoeff)?;
        let source_binding = register_bindings(ctx.bindings, ctx.values, sampled)?;
        let destination_binding = register_bindings(ctx.bindings, ctx.values, decomposed)?;
        let op = push_preimage_op(
            ctx,
            if dropped.is_some() {
                GpuImplementation::gadget_decompose_coeff()
            } else {
                GpuImplementation::gadget_decompose_small_balanced()
            },
            if let Some(dropped) = dropped {
                Box::new([
                    KernelArg::Value(sampled),
                    KernelArg::U32(0),
                    KernelArg::Value(decomposed),
                    KernelArg::U32(0),
                    KernelArg::U32(control_params.base_bits()),
                    KernelArg::U32(
                        u32::try_from(dropped).map_err(|_| "GPU dropped CRT count exceeds u32")?,
                    ),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(destination_binding),
                ])
            } else {
                Box::new([
                    KernelArg::Value(sampled),
                    KernelArg::U32(0),
                    KernelArg::Value(decomposed),
                    KernelArg::U32(0),
                    KernelArg::U32(control_params.base_bits()),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(destination_binding),
                ])
            },
            Box::new([decomposed]),
            Box::new([sample]),
        )?;
        ctx.producer
            .insert(decomposed, vec![(ColumnRange { start: 0, end: output_matrix.columns }, op)]);
        let ConcreteWireType::SmallMatrix { max_coefficient_bound, .. } = &output_ty else {
            return Err("GPU decomposed hash lost its small matrix type".into());
        };
        let bound =
            max_coefficient_bound.to_biguint().ok_or("GPU decomposed hash bound is negative")?;
        let mut bound_words = bound.to_u64_digits();
        if bound_words.is_empty() {
            bound_words.push(0);
        }
        let (output, _) = allocate_compact_value(ctx, output_ty, true)?;
        let pack_status_owner = Arc::new(
            GpuExportStatus::new(&control_params, ctx.device).map_err(|error| error.to_string())?,
        );
        let pack_status = allocate_preimage_control(
            ctx,
            4,
            BoundStorage::from_export_status(Arc::clone(&pack_status_owner))?,
        )?;
        ctx.control_resets.push(ControlReset::IntegerStatus(pack_status_owner));
        let source_binding = register_bindings(ctx.bindings, ctx.values, decomposed)?;
        let output_binding = register_preimage_control_binding(ctx, output)?;
        let status_binding = register_preimage_control_binding(ctx, pack_status)?;
        let packed = push_preimage_op(
            ctx,
            if dropped.is_some() {
                GpuImplementation::compact_pack()
            } else {
                GpuImplementation::compact_pack_per_crt_limb()
            },
            if dropped.is_some() {
                Box::new([
                    KernelArg::U32(output.0),
                    KernelArg::Value(decomposed),
                    KernelArg::U32(0),
                    KernelArg::Value(output),
                    KernelArg::U32(0),
                    KernelArg::Value(pack_status),
                    KernelArg::U32(0),
                    KernelArg::U64List(bound_words.into_boxed_slice()),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(output_binding),
                    KernelArg::U32(status_binding),
                ])
            } else {
                Box::new([
                    KernelArg::Value(decomposed),
                    KernelArg::U32(0),
                    KernelArg::Value(output),
                    KernelArg::U32(0),
                    KernelArg::Value(pack_status),
                    KernelArg::U32(0),
                    KernelArg::U64(bound.to_u64().ok_or("GPU per-limb bound exceeds u64")?),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(output_binding),
                    KernelArg::U32(status_binding),
                ])
            },
            Box::new([output]),
            Box::new([op]),
        )?;
        ctx.producer
            .insert(output, vec![(ColumnRange { start: 0, end: output_matrix.columns }, packed)]);
        ctx.wire_ids.insert(output_wire, output);
    } else {
        let output = allocate_scratch_matrix(ctx, &output_matrix, PhysicalEncoding::FullEval)?;
        emit_matrix_operation(ctx, GpuImplementation::ntt(false), &[sampled], output)?;
        ctx.wire_ids.insert(output_wire, output);
    }
    Ok(())
}

pub(super) fn lower_trapdoor_sample_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } = node.kind() else {
        return Err("GPU trapdoor lowerer received another node".into());
    };
    let public_wire = WireRef { node: node_id, port: Port(0) };
    let secret_wire = WireRef { node: node_id, port: Port(1) };
    let public_ty = wire_types
        .get(&public_wire)
        .and_then(ConcreteWireType::matrix_type)
        .ok_or_else(|| "GPU trapdoor public output has no matrix type".to_owned())?
        .clone();
    let secret_ty = wire_types
        .get(&secret_wire)
        .ok_or_else(|| "GPU trapdoor secret output has no type".to_owned())?
        .clone();
    if !matches!(&secret_ty, ConcreteWireType::Trapdoor {matrix,..} if matrix==&public_ty) {
        return Err("GPU trapdoor outputs disagree on their ordered ring or shape".into());
    }
    let sigma = sigma
        .evaluate_f64_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| error.to_string())?;
    let base = gadget_base
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| error.to_string())?;
    let digits = digit_count
        .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
        .map_err(|error| error.to_string())?
        .to_usize()
        .ok_or_else(|| "GPU trapdoor digit count exceeds usize".to_owned())?;
    let params = ctx.backend.parameters_on_physical_device(ctx.device, &public_ty)?;
    let depth = public_ty.ring.crt_depth();
    if !sigma.is_finite() ||
        sigma <= 0.0 ||
        base != (BigInt::from(1u8) << params.base_bits()) ||
        digits != params.modulus_digits() ||
        params.dropped_moduli() != 0 ||
        !digits.is_multiple_of(depth) ||
        public_ty.columns !=
            public_ty
                .rows
                .checked_mul(digits + 2)
                .ok_or_else(|| "GPU trapdoor width overflows".to_owned())?
    {
        return Err("GPU trapdoor requires the exact regular gadget layout and sigma".into());
    }
    let d = public_ty.rows;
    let dk = d.checked_mul(digits).ok_or_else(|| "GPU trapdoor R width overflows".to_owned())?;
    let shape = |rows, columns| ConcreteMatrixType { ring: public_ty.ring.clone(), rows, columns };
    let re_ty = shape(d, dk);
    let square_ty = shape(d, d);
    let r = emit_fresh_matrix_sample(ctx, scope_id, node_id, 0, &re_ty, Some(sigma))?;
    let e = emit_fresh_matrix_sample(ctx, scope_id, node_id, 1, &re_ty, Some(sigma))?;
    let a_bar = emit_fresh_matrix_sample(ctx, scope_id, node_id, 2, &square_ty, None)?;
    let a_cov = allocate_scratch_matrix(ctx, &square_ty, PhysicalEncoding::FullEval)?;
    let b_cov = allocate_scratch_matrix(ctx, &square_ty, PhysicalEncoding::FullEval)?;
    let d_cov = allocate_scratch_matrix(ctx, &square_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_mul_transpose_rhs(), &[r, r], a_cov)?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_mul_transpose_rhs(), &[r, e], b_cov)?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_mul_transpose_rhs(), &[e, e], d_cov)?;
    let re = allocate_scratch_matrix(ctx, &shape(2 * d, dk), PhysicalEncoding::FullEval)?;
    let re_top = matrix_view(ctx, re, 0, d, 0, dk)?;
    let re_bottom = matrix_view(ctx, re, d, d, 0, dk)?;
    let top = emit_matrix_operation(ctx, GpuImplementation::matrix_copy_view(), &[r], re_top)?;
    let bottom =
        emit_matrix_operation(ctx, GpuImplementation::matrix_copy_view(), &[e], re_bottom)?;
    ctx.producer.insert(
        re,
        vec![(ColumnRange { start: 0, end: dk }, top), (ColumnRange { start: 0, end: dk }, bottom)],
    );

    let product = allocate_scratch_matrix(ctx, &re_ty, PhysicalEncoding::FullEval)?;
    let sum = allocate_scratch_matrix(ctx, &re_ty, PhysicalEncoding::FullEval)?;
    let gadget = allocate_scratch_matrix(ctx, &re_ty, PhysicalEncoding::FullEval)?;
    let a1 = allocate_scratch_matrix(ctx, &re_ty, PhysicalEncoding::FullEval)?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_mul(false), &[a_bar, r], product)?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_add_sub(false), &[product, e], sum)?;
    let gadget_binding = register_bindings(ctx.bindings, ctx.values, gadget)?;
    let residues = public_ty
        .ring
        .crt_moduli()
        .iter()
        .map(|modulus| {
            let modulus = BigInt::from(*modulus);
            ((&base % &modulus + &modulus) % &modulus)
                .to_u64()
                .ok_or_else(|| "GPU gadget residue exceeds u64".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    emit_matrix_fill(
        ctx,
        gadget,
        GpuImplementation::gadget_fill(),
        Box::new([
            KernelArg::Value(gadget),
            KernelArg::U32(0),
            KernelArg::U64(d as u64),
            KernelArg::U32(
                u32::try_from(digits / depth)
                    .map_err(|_| "GPU digits per CRT tower exceeds u32".to_owned())?,
            ),
            KernelArg::U64List(residues.into_boxed_slice()),
            KernelArg::U64(0),
            KernelArg::U32(gadget_binding),
        ]),
    )?;
    emit_matrix_operation(ctx, GpuImplementation::matrix_add_sub(true), &[gadget, sum], a1)?;

    let public = allocate_scratch_matrix(ctx, &public_ty, PhysicalEncoding::FullEval)?;
    let a_view = matrix_view(ctx, public, 0, d, 0, d)?;
    let i_view = matrix_view(ctx, public, 0, d, d, d)?;
    let a1_view = matrix_view(ctx, public, 0, d, 2 * d, dk)?;
    let a_copy =
        emit_matrix_operation(ctx, GpuImplementation::matrix_copy_view(), &[a_bar], a_view)?;
    let identity_binding = register_bindings(ctx.bindings, ctx.values, i_view)?;
    let identity = emit_matrix_fill(
        ctx,
        i_view,
        GpuImplementation::identity_fill(),
        Box::new([
            KernelArg::Value(i_view),
            KernelArg::U32(0),
            KernelArg::U64(d as u64),
            KernelArg::U64(d as u64),
            KernelArg::U32(identity_binding),
        ]),
    )?;
    let a1_copy =
        emit_matrix_operation(ctx, GpuImplementation::matrix_copy_view(), &[a1], a1_view)?;
    ctx.producer.insert(
        public,
        vec![
            (ColumnRange { start: 0, end: d }, a_copy),
            (ColumnRange { start: d, end: 2 * d }, identity),
            (ColumnRange { start: 2 * d, end: 2 * d + dk }, a1_copy),
        ],
    );
    let secret = pack_trapdoor_leaves(
        ctx,
        secret_ty,
        [r, e, a_cov, b_cov, d_cov, re],
        PhysicalEncoding::FullEval,
    )?;
    ctx.wire_ids.insert(public_wire, public);
    ctx.wire_ids.insert(secret_wire, secret);
    ctx.trapdoor_public_ids.insert(secret, public);
    Ok(())
}

fn lower_public_gadget_preimage(
    ctx: &mut PhysicalLoweringContext<'_>,
    output_wire: WireRef,
    output_ty: ConcreteWireType,
    public_ty: &ConcreteMatrixType,
    target_ty: &ConcreteMatrixType,
    target: PhysicalValueId,
    gadget_base: &BigInt,
    digit_count: usize,
) -> Result<(), String> {
    let ConcreteWireType::Preimage { matrix: output_matrix, max_coefficient_bound, bound_domain } =
        &output_ty
    else {
        return Err("GPU gadget preimage output is not bounded".into());
    };
    if *bound_domain != CoefficientBoundDomain::Global {
        return Err("GPU regular gadget preimage requires a globally bounded output".into());
    }
    let output_matrix = output_matrix.clone();
    let max_coefficient_bound = max_coefficient_bound.clone();
    if public_ty.ring != target_ty.ring ||
        public_ty.rows != target_ty.rows ||
        public_ty.columns != output_matrix.rows ||
        target_ty.columns != output_matrix.columns ||
        output_matrix.ring != target_ty.ring
    {
        return Err("GPU gadget preimage dimensions or ordered ring disagree".into());
    }
    let params = ctx.backend.parameters_on_physical_device(ctx.device, target_ty)?;
    let base_bits = params.base_bits();
    if *gadget_base != (BigInt::from(1u8) << base_bits) ||
        digit_count != params.modulus_digits() ||
        params.gadget_dropped_moduli(Some(digit_count)) != Some(0)
    {
        return Err("GPU exact public gadget preimage needs full regular CRT gadget".into());
    }
    let source_coefficient = match ctx.values[target.0 as usize].encodings.as_ref() {
        [PhysicalEncoding::FullCoeff] => target,
        [PhysicalEncoding::FullEval] => {
            let coefficient = allocate_scratch_matrix(ctx, target_ty, PhysicalEncoding::FullCoeff)?;
            emit_matrix_operation(ctx, GpuImplementation::ntt(true), &[target], coefficient)?;
            coefficient
        }
        _ => return Err("GPU gadget preimage target is not a full matrix".into()),
    };
    let candidate = allocate_scratch_matrix(ctx, &output_matrix, PhysicalEncoding::FullCoeff)?;
    let source_binding = register_bindings(ctx.bindings, ctx.values, source_coefficient)?;
    let candidate_binding = register_bindings(ctx.bindings, ctx.values, candidate)?;
    let decompose = push_preimage_op(
        ctx,
        GpuImplementation::gadget_decompose_coeff(),
        Box::new([
            KernelArg::Value(source_coefficient),
            KernelArg::U32(0),
            KernelArg::Value(candidate),
            KernelArg::U32(0),
            KernelArg::U32(base_bits),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(candidate_binding),
        ]),
        Box::new([candidate]),
        all_predecessors(ctx.producer, source_coefficient),
    )?;
    ctx.producer
        .insert(candidate, vec![(ColumnRange { start: 0, end: output_matrix.columns }, decompose)]);
    let (output, _) = allocate_compact_value(ctx, output_ty, true)?;
    let control_params = ctx.backend.control_parameters_on_device(ctx.device)?;
    let status_owner = Arc::new(
        GpuExportStatus::new(&control_params, ctx.device).map_err(|error| error.to_string())?,
    );
    let status = allocate_preimage_control(
        ctx,
        4,
        BoundStorage::from_export_status(Arc::clone(&status_owner))?,
    )?;
    ctx.control_resets.push(ControlReset::IntegerStatus(status_owner));
    let mut bound_words = max_coefficient_bound
        .to_biguint()
        .ok_or("GPU gadget preimage bound is negative")?
        .to_u64_digits();
    if bound_words.is_empty() {
        bound_words.push(0);
    }
    let output_binding = register_preimage_control_binding(ctx, output)?;
    let status_binding = register_preimage_control_binding(ctx, status)?;
    let packed = push_preimage_op(
        ctx,
        GpuImplementation::compact_pack(),
        Box::new([
            KernelArg::U32(output.0),
            KernelArg::Value(candidate),
            KernelArg::U32(0),
            KernelArg::Value(output),
            KernelArg::U32(0),
            KernelArg::Value(status),
            KernelArg::U32(0),
            KernelArg::U64List(bound_words.into_boxed_slice()),
            KernelArg::U32(candidate_binding),
            KernelArg::U32(output_binding),
            KernelArg::U32(status_binding),
        ]),
        Box::new([output]),
        Box::new([decompose]),
    )?;
    ctx.producer
        .insert(output, vec![(ColumnRange { start: 0, end: output_matrix.columns }, packed)]);
    ctx.wire_ids.insert(output_wire, output);
    Ok(())
}

/// Copy a device's compact column block into the `block` columns of the
/// home owner `destination`, one contiguous copy per row of the row-major
/// payload, after the block's writers.
fn copy_compact_block_home(
    ctx: &mut PhysicalLoweringContext<'_>,
    source: PhysicalValueId,
    destination: PhysicalValueId,
    block: ColumnRange,
) -> Result<Vec<u32>, String> {
    let implementation =
        ctx.implementations.register(GpuImplementation::copy()).map_err(str::to_owned)?;
    let predecessors = all_predecessors(ctx.producer, source);
    let rows = ctx.values[source.0 as usize].parts[0].view.extent[0];
    let mut copies = Vec::new();
    for row in 0..rows {
        let mut window =
            |id: PhysicalValueId, column: usize| -> Result<(PhysicalValueId, u64), String> {
                let mut physical = ctx.values[id.0 as usize].clone();
                let [part] = physical.parts.as_mut() else {
                    return Err("GPU compact block needs one physical part".into());
                };
                let view = &mut part.view;
                let column = u64::try_from(column).map_err(|_| "GPU column exceeds u64")?;
                view.byte_offset += row * view.byte_strides[0] + column * view.byte_strides[1];
                view.origin[0] = row;
                view.extent[0] = 1;
                view.origin[1] = column;
                view.extent[1] = (block.end - block.start) as u64;
                let bytes = view.extent[1] * view.byte_strides[1];
                let owner = ctx.owners.get(&id).ok_or("GPU compact block owner is absent")?;
                let resident =
                    owner.with_physical_view(Arc::new(physical.clone())).map_err(str::to_owned)?;
                let window = value_id(ctx.values.len())?;
                ctx.values.push(physical);
                ctx.owners.insert(window, Arc::new(resident));
                Ok((window, bytes))
            };
        let (from, bytes) = window(source, 0)?;
        let (to, _) = window(destination, block.start)?;
        let binding =
            u32::try_from(ctx.bindings.len()).map_err(|_| "too many GPU graph bindings")?;
        ctx.bindings.push(GpuBindingSource::PhysicalPart { value: from, part: 0, limb: 0 });
        ctx.bindings.push(GpuBindingSource::PhysicalPart { value: to, part: 0, limb: 0 });
        copies.push(u32::try_from(ctx.operations.len()).map_err(|_| "too many GPU operations")?);
        ctx.operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(from),
                KernelArg::U32(0),
                KernelArg::Value(to),
                KernelArg::U32(0),
                KernelArg::U64(bytes),
                KernelArg::U32(binding),
                KernelArg::U32(binding + 1),
            ]),
            outputs: Box::new([to]),
            device: ctx.device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors: predecessors.clone(),
            body: None,
        });
    }
    Ok(copies)
}

pub(super) fn lower_preimage_sample_node(
    ctx: &mut PhysicalLoweringContext<'_>,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::types::NodeId,
    node: &mxx_ir_core::graph::NodeHandle,
    env: &ParamEnv,
    wire_types: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<(), String> {
    let NodeKind::PreimageSample { .. } = node.kind() else {
        return Err("GPU preimage lowerer received another node".into());
    };
    let scope = ctx
        .validated
        .source
        .scope(scope_id)
        .ok_or_else(|| "GPU preimage scope is missing".to_owned())?;
    let arguments =
        scope.arguments(node).ok_or_else(|| "GPU preimage arguments are missing".to_owned())?;
    let [public_wire, secret_wire, target_wire] = arguments.as_slice() else {
        return Err("GPU preimage needs public, trapdoor, and target".into());
    };
    let public = *ctx.wire_ids.get(public_wire).ok_or("GPU preimage public input is absent")?;
    let secret = *ctx.wire_ids.get(secret_wire).ok_or("GPU preimage trapdoor input is absent")?;
    let target = *ctx.wire_ids.get(target_wire).ok_or("GPU preimage target input is absent")?;
    if ctx.trapdoor_public_ids.get(&secret).is_some_and(|paired| *paired != public) {
        return Err("GPU preimage public input is not paired with its trapdoor".into());
    }
    let public_ty = ctx.values[public.0 as usize]
        .ty
        .matrix_type()
        .ok_or("GPU preimage public input is not a matrix")?
        .clone();
    let target_ty = ctx.values[target.0 as usize]
        .ty
        .matrix_type()
        .ok_or("GPU preimage target input is not a matrix")?
        .clone();
    let ConcreteWireType::Trapdoor {
        matrix: trapdoor_matrix, sigma, gadget_base, digit_count, ..
    } = &ctx.values[secret.0 as usize].ty
    else {
        return Err("GPU preimage secret input is not a typed trapdoor".into());
    };
    if *trapdoor_matrix != public_ty ||
        target_ty.ring != public_ty.ring ||
        target_ty.rows != public_ty.rows
    {
        return Err("GPU preimage inputs disagree on ring or matrix shape".into());
    }
    let sigma = sigma.clone();
    let gadget_base = gadget_base.clone();
    let digit_count = *digit_count;
    let output_wire = WireRef { node: node_id, port: Port(0) };
    let output_ty = wire_types
        .get(&output_wire)
        .ok_or_else(|| "GPU preimage output has no validated type".to_owned())?
        .clone();
    let ConcreteWireType::Preimage { matrix: output_matrix, bound_domain, .. } = &output_ty else {
        return Err("GPU preimage output is not bounded".into());
    };
    if ctx.values[secret.0 as usize].encodings.as_ref() == [PhysicalEncoding::PublicGadgetEval] {
        return lower_public_gadget_preimage(
            ctx,
            output_wire,
            output_ty,
            &public_ty,
            &target_ty,
            target,
            &gadget_base,
            digit_count,
        );
    }
    if *bound_domain != CoefficientBoundDomain::Global {
        return Err("GPU sampled trapdoor preimage needs a globally bounded output".into());
    }
    let sigma = sigma.evaluate_f64(env).map_err(|error| error.to_string())?;
    let output_matrix = output_matrix.clone();
    let params = ctx.backend.parameters_on_physical_device(ctx.device, &public_ty)?;
    let d = public_ty.rows;
    let t = target_ty.columns;
    let digits = digit_count;
    let dk = d.checked_mul(digits).ok_or("GPU preimage gadget width overflows")?;
    if !sigma.is_finite() ||
        sigma <= 0.0 ||
        gadget_base != (BigInt::from(1u8) << params.base_bits()) ||
        digits != params.modulus_digits() ||
        params.dropped_moduli() != 0 ||
        public_ty.columns != d.checked_mul(digits + 2).ok_or("GPU preimage width overflows")? ||
        output_matrix.ring != public_ty.ring ||
        output_matrix.rows != public_ty.columns ||
        output_matrix.columns != t ||
        t == 0
    {
        return Err("GPU preimage requires exact regular trapdoor parameters and shapes".into());
    }
    let choices = ctx
        .logical
        .nodes
        .iter()
        .filter(|choice| {
            choice.key.site == node_id.0 &&
                choice.key.shape_class ==
                    scope_shape_class(ctx.validated, scope_id).unwrap_or(u64::MAX) &&
                choice.key.instance_class == 0
        })
        .collect::<Vec<_>>();
    let [choice] = choices.as_slice() else {
        return Err("GPU preimage has no unique frozen choice".into());
    };
    let max_attempts = choice
        .preimage_max_attempts
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0 && *value < u32::MAX)
        .ok_or("GPU preimage retry bound is absent or too large")?;
    let layout = ctx
        .logical
        .layouts
        .iter()
        .find(|layout| choice.output_layouts.first() == Some(&layout.id))
        .ok_or("GPU preimage output layout is absent from the frozen plan")?;
    if layout.columns != output_matrix.columns ||
        choice.columns_per_job.iter().all(|width| *width == 0)
    {
        return Err("GPU preimage tile width or layout differs from the frozen plan".into());
    }
    // Tiles split the output columns into one contiguous block per device.
    // A remote tile samples next to copies of the trapdoor and its target
    // window, and its device's block is copied home row by row.
    let devices = ctx
        .logical
        .contract
        .logical_to_physical_devices
        .iter()
        .map(|&device| i32::try_from(device).map_err(|_| "GPU device ID overflows".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    let home = ctx.device;
    let tiles = layout
        .schedule(&choice.columns_per_job, 0)
        .map_err(|error| error.to_string())?
        .waves()
        .flatten()
        .map(|job| {
            let device = if ctx.device_body || home != devices[0] {
                home
            } else {
                *devices.get(job.device).ok_or("GPU preimage tile names an unknown device")?
            };
            Ok((device, job.start, job.end - job.start))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut blocks = BTreeMap::<i32, ColumnRange>::new();
    for &(device, start, width) in tiles.iter().filter(|tile| tile.0 != home) {
        let block = blocks.entry(device).or_insert(ColumnRange { start, end: start + width });
        *block = ColumnRange { start: block.start.min(start), end: block.end.max(start + width) };
    }
    let c = (gadget_base.to_f64().ok_or("GPU gadget base exceeds f64")? + 1.0) * sigma;
    let n = public_ty.ring.ring_dimension() as usize;
    let s = 1.8 *
        (gadget_base.to_f64().ok_or("GPU gadget base exceeds f64")? + 1.0) *
        sigma *
        sigma *
        (((d * n * digits) as f64).sqrt() + ((2 * n) as f64).sqrt() + 4.7);
    let dgg_stddev = (s * s - c * c).sqrt();
    if !c.is_finite() || !s.is_finite() || !dgg_stddev.is_finite() || !(s > c) || dgg_stddev <= 0.0
    {
        return Err("GPU preimage covariance parameters are invalid".into());
    }
    let shape = |rows, columns| ConcreteMatrixType { ring: public_ty.ring.clone(), rows, columns };
    let r = trapdoor_leaf_view(ctx, secret, 0)?;
    let e = trapdoor_leaf_view(ctx, secret, 1)?;
    let re = trapdoor_leaf_view(ctx, secret, 5)?;
    let secret_coeff = inverse_trapdoor_for_export(ctx, secret)?;
    let depth = public_ty.ring.crt_depth();
    let secret_coeff_bindings = register_all_parts(ctx.bindings, ctx.values, secret_coeff)?;
    let (full_output, magnitude_bytes) = allocate_compact_value(ctx, output_ty.clone(), false)?;
    // Per remote device: its copies of the trapdoor inputs, the coefficient
    // bindings of its inverse trapdoor, and its output block.
    let mut remote = BTreeMap::new();
    for (&device, block) in &blocks {
        let copies = [public, r, e, re, secret_coeff]
            .into_iter()
            .map(|id| replicate_to_device(ctx, id, device))
            .collect::<Result<Vec<_>, _>>()?;
        let coeff_bindings = register_all_parts(ctx.bindings, ctx.values, copies[4])?;
        let mut block_ty = output_ty.clone();
        if let ConcreteWireType::Preimage { matrix, .. } = &mut block_ty {
            matrix.columns = block.end - block.start;
        }
        ctx.device = device;
        let (block_output, _) = allocate_compact_value(ctx, block_ty, false)?;
        ctx.device = home;
        remote.insert(device, (copies, coeff_bindings, block_output, *block));
    }
    for &(tile_device, tile_start, tile_width) in &tiles {
        let target_source = target;
        let target = matrix_view(ctx, target_source, 0, d, tile_start, tile_width)?;
        let target_predecessors = all_predecessors(ctx.producer, target_source);
        ctx.producer.insert(
            target,
            target_predecessors
                .iter()
                .map(|&op| (ColumnRange { start: 0, end: tile_width }, op))
                .collect(),
        );
        let output_matrix = shape(output_matrix.rows, tile_width);
        let t = tile_width;
        // Matrix arithmetic requires every operand in a tile to use the same
        // local origin. The source window addresses the owner's real column
        // through its checked byte offset; copy it once into a tile-shaped
        // value whose origin is zero before the retry Graph reads it.
        let tile_target_ty = shape(d, t);
        let mut target_local =
            allocate_scratch_matrix(ctx, &tile_target_ty, PhysicalEncoding::FullEval)?;
        emit_matrix_operation(ctx, GpuImplementation::matrix_copy_view(), &[target], target_local)?;
        let (public, r, e, re, secret_coeff, secret_coeff_bindings, tile_owner, tile_column) =
            match remote.get(&tile_device) {
                Some((copies, coeff_bindings, block_output, block)) => {
                    target_local = replicate_to_device(ctx, target_local, tile_device)?;
                    ctx.device = tile_device;
                    let [public, r, e, re, secret_coeff] = copies[..] else {
                        return Err("GPU preimage device inputs are incomplete".into());
                    };
                    let column = tile_start - block.start;
                    (public, r, e, re, secret_coeff, *coeff_bindings, *block_output, column)
                }
                None => {
                    (public, r, e, re, secret_coeff, secret_coeff_bindings, full_output, tile_start)
                }
            };
        let output = compact_column_view(ctx, tile_owner, tile_column, tile_width)?;
        let resource_base = u32::try_from(ctx.values.len())
            .ok()
            .and_then(|value| value.checked_mul(3))
            .ok_or("GPU preimage resource ID overflows")?;
        let p1_resource = resource_base;
        let gq_resource = resource_base + 1;
        let cutoff_resource = resource_base + 2;
        let p1_workspace = register_preimage_workspace(
            ctx.bindings,
            GpuPreparedWorkspaceKind::P1,
            p1_resource,
            5,
        )?;
        let refresh = push_preimage_op(
            ctx,
            GpuImplementation::p1_covariance_refresh(),
            Box::new([
                KernelArg::U32(p1_resource),
                KernelArg::Value(secret_coeff),
                KernelArg::U32((2 * depth) as u32),
                KernelArg::Value(secret_coeff),
                KernelArg::U32((3 * depth) as u32),
                KernelArg::Value(secret_coeff),
                KernelArg::U32((4 * depth) as u32),
                KernelArg::F64(c),
                KernelArg::F64(s),
                KernelArg::F64(dgg_stddev),
                KernelArg::U64(t as u64),
                KernelArg::U32(secret_coeff_bindings + (2 * depth) as u32),
                KernelArg::U32(secret_coeff_bindings + (3 * depth) as u32),
                KernelArg::U32(secret_coeff_bindings + (4 * depth) as u32),
                KernelArg::U32(p1_workspace),
                KernelArg::U32(p1_workspace + 1),
                KernelArg::U32(p1_workspace + 2),
                KernelArg::U32(p1_workspace + 3),
                KernelArg::U32(p1_workspace + 4),
            ]),
            Box::new([]),
            all_predecessors(ctx.producer, secret_coeff),
        )?;
        let control_params = ctx.backend.control_parameters_on_device(ctx.device)?;
        let attempt = Arc::new(
            GpuPreimageAttempt::new(&control_params, ctx.device)
                .map_err(|error| error.to_string())?,
        );
        let status = Arc::new(
            GpuPreimageStatus::new(&control_params, ctx.device)
                .map_err(|error| error.to_string())?,
        );
        let limit = Arc::new(
            GpuSignedValues::from_canonical_u64(
                &control_params,
                ctx.device,
                &[max_attempts as u64],
            )
            .map_err(|error| error.to_string())?,
        );
        let attempt_id = allocate_preimage_control(
            ctx,
            8,
            BoundStorage::from_preimage_attempt(Arc::clone(&attempt))?,
        )?;
        let status_id = allocate_preimage_control(
            ctx,
            16,
            BoundStorage::from_preimage_status(Arc::clone(&status))?,
        )?;
        let limit_id = allocate_preimage_control(ctx, 8, BoundStorage::from_signed_values(limit)?)?;
        // The retry WHILE continues only while the status latch word
        // (`MxxPreimageStatus::reserved`, byte 12) is zero, so the first
        // accepted attempt ends the loop instead of resampling until the bound.
        let latch_id = {
            let mut latch = ctx.values[status_id.0 as usize].clone();
            latch.ty = ConcreteWireType::Bytes { length: 4 };
            latch.parts[0].view.byte_offset = 12;
            latch.parts[0].view.extent = Box::new([4]);
            let owner = ctx.owners.get(&status_id).ok_or("GPU preimage status owner is absent")?;
            let resident =
                owner.with_physical_view(Arc::new(latch.clone())).map_err(str::to_owned)?;
            let id = value_id(ctx.values.len())?;
            ctx.values.push(latch);
            ctx.owners.insert(id, Arc::new(resident));
            id
        };
        ctx.preimage_replays.push(PreimageReplay { attempt, status, planned_max: max_attempts });
        let gq_workspace = register_preimage_workspace(
            ctx.bindings,
            GpuPreparedWorkspaceKind::Gq,
            gq_resource,
            1,
        )?;
        let cutoff_staging = register_preimage_workspace(
            ctx.bindings,
            GpuPreparedWorkspaceKind::Cutoff,
            cutoff_resource,
            1,
        )?;
        let attempt_binding = register_preimage_control_binding(ctx, attempt_id)?;
        let status_binding = register_preimage_control_binding(ctx, status_id)?;
        let output_binding = register_preimage_control_binding(ctx, output)?;
        let limit_binding = register_preimage_control_binding(ctx, limit_id)?;
        let latch_binding = register_preimage_control_binding(ctx, latch_id)?;
        let mut body = Vec::<CompiledGpuOp>::new();
        let mut body_producer = BTreeMap::<PhysicalValueId, Vec<(ColumnRange, u32)>>::new();
        {
            let mut child = PhysicalLoweringContext {
                validated: ctx.validated,
                integer_input_ranges: ctx.integer_input_ranges,
                artifact_payload_sizes: ctx.artifact_payload_sizes,
                backend: ctx.backend,
                logical: ctx.logical,
                device: ctx.device,
                values: &mut *ctx.values,
                owners: &mut *ctx.owners,
                wire_ids: &mut *ctx.wire_ids,
                implementations: &mut *ctx.implementations,
                operations: &mut body,
                bindings: &mut *ctx.bindings,
                producer: &mut body_producer,
                family_member_producers: &mut *ctx.family_member_producers,
                control_resets: &mut *ctx.control_resets,
                sample_seeds: &mut *ctx.sample_seeds,
                hash_resources: &mut *ctx.hash_resources,
                real_owners: &mut *ctx.real_owners,
                indexed_tables: &mut *ctx.indexed_tables,
                dynamic_export_resources: &mut *ctx.dynamic_export_resources,
                device_loop_indices: ctx.device_loop_indices.clone(),
                lanes: ctx.lanes,
                active_parallel_template: ctx.active_parallel_template,
                active_parallel_instances: ctx.active_parallel_instances.clone(),
                device_body: true,
                preimage_replays: &mut *ctx.preimage_replays,
                trapdoor_public_ids: &mut *ctx.trapdoor_public_ids,
                waves: &mut *ctx.waves,
                import_templates: &mut *ctx.import_templates,
                external_io_loops: &mut *ctx.external_io_loops,
                external_io_imports: &mut *ctx.external_io_imports,
                crt_resource_next: &mut *ctx.crt_resource_next,
                converted: &mut BTreeMap::new(),
                integer_status: &mut *ctx.integer_status,
            };
            let p2_seed = derive_preimage_stage_seed(&mut child, scope_id, node_id, attempt_id, 0)?;
            let (p2_coeff, p2_eval) = emit_fresh_matrix_sample_parts(
                &mut child,
                scope_id,
                node_id,
                3,
                &shape(dk, t),
                Some(dgg_stddev),
                Some(p2_seed),
            )?;
            let tp2_eval =
                allocate_scratch_matrix(&mut child, &shape(2 * d, t), PhysicalEncoding::FullEval)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::matrix_mul(false),
                &[re, p2_eval],
                tp2_eval,
            )?;
            let tp2_coeff =
                allocate_scratch_matrix(&mut child, &shape(2 * d, t), PhysicalEncoding::FullCoeff)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::ntt(true),
                &[tp2_eval],
                tp2_coeff,
            )?;
            let p1_coeff =
                allocate_scratch_matrix(&mut child, &shape(2 * d, t), PhysicalEncoding::FullCoeff)?;
            let p1_seed = derive_preimage_stage_seed(&mut child, scope_id, node_id, attempt_id, 1)?;
            let tp2_binding = register_bindings(child.bindings, child.values, tp2_coeff)?;
            let p1_binding = register_bindings(child.bindings, child.values, p1_coeff)?;
            let p1_seed_binding = register_preimage_control_binding(&mut child, p1_seed)?;
            let tp2_predecessors = all_predecessors(child.producer, tp2_coeff);
            let p1_index = push_preimage_op(
                &mut child,
                GpuImplementation::p1_sample(),
                Box::new([
                    KernelArg::U32(p1_resource),
                    KernelArg::Value(tp2_coeff),
                    KernelArg::U32(0),
                    KernelArg::Value(p1_coeff),
                    KernelArg::U32(0),
                    KernelArg::Value(p1_seed),
                    KernelArg::U32(0),
                    KernelArg::F64(c),
                    KernelArg::F64(s),
                    KernelArg::U32(tp2_binding),
                    KernelArg::U32(p1_binding),
                    KernelArg::U32(p1_seed_binding),
                    KernelArg::U32(p1_workspace),
                    KernelArg::U32(p1_workspace + 1),
                    KernelArg::U32(p1_workspace + 2),
                    KernelArg::U32(p1_workspace + 3),
                    KernelArg::U32(p1_workspace + 4),
                ]),
                Box::new([p1_coeff]),
                tp2_predecessors,
            )?;
            child.producer.insert(p1_coeff, vec![(ColumnRange { start: 0, end: t }, p1_index)]);
            let perturb_coeff =
                allocate_scratch_matrix(&mut child, &output_matrix, PhysicalEncoding::FullCoeff)?;
            let p1_dest = matrix_view(&mut child, perturb_coeff, 0, 2 * d, 0, t)?;
            let p2_dest = matrix_view(&mut child, perturb_coeff, 2 * d, dk, 0, t)?;
            let p1_copy = emit_matrix_operation(
                &mut child,
                GpuImplementation::matrix_copy_view(),
                &[p1_coeff],
                p1_dest,
            )?;
            let p2_copy = emit_matrix_operation(
                &mut child,
                GpuImplementation::matrix_copy_view(),
                &[p2_coeff],
                p2_dest,
            )?;
            child.producer.insert(
                perturb_coeff,
                vec![
                    (ColumnRange { start: 0, end: t }, p1_copy),
                    (ColumnRange { start: 0, end: t }, p2_copy),
                ],
            );
            let perturb_eval =
                allocate_scratch_matrix(&mut child, &output_matrix, PhysicalEncoding::FullEval)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::ntt(false),
                &[perturb_coeff],
                perturb_eval,
            )?;
            let product_eval =
                allocate_scratch_matrix(&mut child, &tile_target_ty, PhysicalEncoding::FullEval)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::matrix_mul(false),
                &[public, perturb_eval],
                product_eval,
            )?;
            let residual_eval =
                allocate_scratch_matrix(&mut child, &tile_target_ty, PhysicalEncoding::FullEval)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::matrix_add_sub(true),
                &[target_local, product_eval],
                residual_eval,
            )?;
            let residual_coeff =
                allocate_scratch_matrix(&mut child, &tile_target_ty, PhysicalEncoding::FullCoeff)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::ntt(true),
                &[residual_eval],
                residual_coeff,
            )?;
            let z_coeff =
                allocate_scratch_matrix(&mut child, &shape(dk, t), PhysicalEncoding::FullCoeff)?;
            let gq_seed = derive_preimage_stage_seed(&mut child, scope_id, node_id, attempt_id, 2)?;
            let residual_binding = register_bindings(child.bindings, child.values, residual_coeff)?;
            let z_binding = register_bindings(child.bindings, child.values, z_coeff)?;
            let gq_seed_binding = register_preimage_control_binding(&mut child, gq_seed)?;
            let residual_predecessors = all_predecessors(child.producer, residual_coeff);
            let gq_index = push_preimage_op(
                &mut child,
                GpuImplementation::gq_sample(),
                Box::new([
                    KernelArg::U32(gq_resource),
                    KernelArg::Value(residual_coeff),
                    KernelArg::U32(0),
                    KernelArg::Value(z_coeff),
                    KernelArg::U32(0),
                    KernelArg::Value(gq_seed),
                    KernelArg::U32(0),
                    KernelArg::U32(params.base_bits()),
                    KernelArg::F64(c),
                    KernelArg::U32(residual_binding),
                    KernelArg::U32(z_binding),
                    KernelArg::U32(gq_seed_binding),
                    KernelArg::U32(gq_workspace),
                ]),
                Box::new([z_coeff]),
                residual_predecessors,
            )?;
            child.producer.insert(z_coeff, vec![(ColumnRange { start: 0, end: t }, gq_index)]);
            let z_eval =
                allocate_scratch_matrix(&mut child, &shape(dk, t), PhysicalEncoding::FullEval)?;
            emit_matrix_operation(&mut child, GpuImplementation::ntt(false), &[z_coeff], z_eval)?;
            let candidate_eval =
                allocate_scratch_matrix(&mut child, &output_matrix, PhysicalEncoding::FullEval)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::matrix_copy_view(),
                &[perturb_eval],
                candidate_eval,
            )?;
            let candidate_binding =
                register_bindings(child.bindings, child.values, candidate_eval)?;
            let r_binding = register_bindings(child.bindings, child.values, r)?;
            let e_binding = register_bindings(child.bindings, child.values, e)?;
            let z_eval_binding = register_bindings(child.bindings, child.values, z_eval)?;
            let mut correction_predecessors = all_predecessors(child.producer, z_eval).into_vec();
            correction_predecessors.extend(all_predecessors(child.producer, candidate_eval));
            correction_predecessors.sort_unstable();
            correction_predecessors.dedup();
            let correction = push_preimage_op(
                &mut child,
                GpuImplementation::preimage_correction(),
                Box::new([
                    KernelArg::Value(candidate_eval),
                    KernelArg::U32(0),
                    KernelArg::Value(r),
                    KernelArg::U32(0),
                    KernelArg::Value(e),
                    KernelArg::U32(0),
                    KernelArg::Value(z_eval),
                    KernelArg::U32(0),
                    KernelArg::U32(candidate_binding),
                    KernelArg::U32(r_binding),
                    KernelArg::U32(e_binding),
                    KernelArg::U32(z_eval_binding),
                ]),
                Box::new([candidate_eval]),
                correction_predecessors.into_boxed_slice(),
            )?;
            child
                .producer
                .insert(candidate_eval, vec![(ColumnRange { start: 0, end: t }, correction)]);
            let candidate_coeff =
                allocate_scratch_matrix(&mut child, &output_matrix, PhysicalEncoding::FullCoeff)?;
            emit_matrix_operation(
                &mut child,
                GpuImplementation::ntt(true),
                &[candidate_eval],
                candidate_coeff,
            )?;
            let candidate_coeff_binding =
                register_bindings(child.bindings, child.values, candidate_coeff)?;
            let bound = match &child.values[output.0 as usize].ty {
                ConcreteWireType::Preimage { max_coefficient_bound, .. } => {
                    max_coefficient_bound.to_biguint().ok_or("GPU preimage bound is negative")?
                }
                _ => return Err("GPU preimage compact output changed type".into()),
            };
            let mut bound_words = bound.to_u64_digits();
            if bound_words.is_empty() {
                bound_words.push(0);
            }
            let candidate_predecessors = all_predecessors(child.producer, candidate_coeff);
            let cutoff = push_preimage_op(
                &mut child,
                GpuImplementation::preimage_cutoff(),
                Box::new([
                    KernelArg::U32(cutoff_resource),
                    KernelArg::Value(candidate_coeff),
                    KernelArg::U32(0),
                    KernelArg::Value(attempt_id),
                    KernelArg::U32(0),
                    KernelArg::Value(status_id),
                    KernelArg::U32(0),
                    KernelArg::U64List(bound_words.into_boxed_slice()),
                    KernelArg::U32(
                        u32::try_from(magnitude_bytes)
                            .map_err(|_| "GPU preimage magnitude width exceeds u32")?,
                    ),
                    KernelArg::U32(candidate_coeff_binding),
                    KernelArg::U32(cutoff_staging),
                    KernelArg::U32(attempt_binding),
                    KernelArg::U32(status_binding),
                ]),
                Box::new([status_id]),
                candidate_predecessors,
            )?;
            let publish = push_preimage_op(
                &mut child,
                GpuImplementation::preimage_publish(),
                Box::new([
                    KernelArg::U32(cutoff_resource),
                    KernelArg::Value(output),
                    KernelArg::U32(0),
                    KernelArg::Value(status_id),
                    KernelArg::U32(0),
                    KernelArg::U32(output_binding),
                    KernelArg::U32(cutoff_staging),
                    KernelArg::U32(status_binding),
                ]),
                Box::new([output]),
                Box::new([cutoff]),
            )?;
            if publish == 0 {
                return Err("GPU preimage body has no sampling work".into());
            }
            for index in 1..body.len() {
                body[index].predecessors =
                    Box::new([u32::try_from(index - 1)
                        .map_err(|_| "GPU preimage body index exceeds u32")?]);
            }
        }
        let mut predecessors = all_predecessors(ctx.producer, public).into_vec();
        predecessors.extend(all_predecessors(ctx.producer, secret));
        predecessors.extend(all_predecessors(ctx.producer, target));
        predecessors.push(refresh);
        predecessors.sort_unstable();
        predecessors.dedup();
        let loop_op = push_preimage_op(
            ctx,
            GpuImplementation::loop_while(),
            Box::new([
                KernelArg::Value(attempt_id),
                KernelArg::U32(0),
                KernelArg::Value(limit_id),
                KernelArg::U32(0),
                KernelArg::Value(latch_id),
                KernelArg::U32(0),
                KernelArg::U64(max_attempts as u64),
                KernelArg::U32(attempt_binding),
                KernelArg::U32(limit_binding),
                KernelArg::U32(latch_binding),
            ]),
            Box::new([]),
            predecessors.into_boxed_slice(),
        )?;
        ctx.operations[loop_op as usize].body = Some(body.into_boxed_slice());
        ctx.producer.insert(output, vec![(ColumnRange { start: 0, end: t }, loop_op)]);
        ctx.producer
            .entry(tile_owner)
            .or_default()
            .push((ColumnRange { start: tile_column, end: tile_column + t }, loop_op));
        ctx.device = home;
    }
    for (_, _, block_output, block) in remote.into_values() {
        let copies = copy_compact_block_home(ctx, block_output, full_output, block)?;
        ctx.producer
            .entry(full_output)
            .or_default()
            .extend(copies.into_iter().map(|copy| (block, copy)));
    }
    let (returned, _) = allocate_compact_value(ctx, output_ty, true)?;
    let source_bytes = ctx.values[full_output.0 as usize]
        .parts
        .first()
        .ok_or("GPU preimage packed source has no physical part")?
        .view
        .extent
        .iter()
        .try_fold(1u64, |size, extent| size.checked_mul(*extent))
        .ok_or("GPU preimage packed copy size overflows")?;
    let source_binding = register_preimage_control_binding(ctx, full_output)?;
    let returned_binding = register_preimage_control_binding(ctx, returned)?;
    let implementation =
        ctx.implementations.register(GpuImplementation::copy()).map_err(str::to_owned)?;
    let copy_index = u32::try_from(ctx.operations.len()).map_err(|_| "too many GPU operations")?;
    ctx.operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(full_output),
            KernelArg::U32(0),
            KernelArg::Value(returned),
            KernelArg::U32(0),
            KernelArg::U64(source_bytes),
            KernelArg::U32(source_binding),
            KernelArg::U32(returned_binding),
        ]),
        outputs: Box::new([returned]),
        device: ctx.device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: all_predecessors(ctx.producer, full_output),
        body: None,
    });
    ctx.producer
        .insert(returned, vec![(ColumnRange { start: 0, end: target_ty.columns }, copy_index)]);
    ctx.wire_ids.insert(output_wire, returned);
    Ok(())
}

fn activate_matrix_import(
    wire: WireRef,
    pending: &mut BTreeMap<WireRef, (ImportTemplate, PhysicalValueId)>,
    values: &[PhysicalValue],
    implementations: &mut GpuImplementationRegistry,
    operations: &mut Vec<CompiledGpuOp>,
    bindings: &mut Vec<GpuBindingSource>,
    producer: &mut BTreeMap<PhysicalValueId, Vec<(ColumnRange, u32)>>,
    templates: &mut Vec<ImportTemplate>,
    device: i32,
) -> Result<(), String> {
    let Some((mut import, eval)) = pending.remove(&wire) else {
        return Ok(());
    };
    if matches!(import.upload_owner, ImportDestination::Trapdoor { .. }) {
        templates.push(import);
        return Ok(());
    }
    if matches!(
        values[import.destination.0 as usize].encodings.as_ref(),
        [PhysicalEncoding::Bytes |
            PhysicalEncoding::TypedBlobLengthPrefixed |
            PhysicalEncoding::Signed(_) |
            PhysicalEncoding::CompactCoeff { .. } |
            PhysicalEncoding::CompactCoeffPerCrtLimb { .. }]
    ) {
        import.before_operation =
            u32::try_from(operations.len()).map_err(|_| "too many GPU operations".to_owned())?;
        templates.push(import);
        return Ok(());
    }
    let source_binding = register_bindings(bindings, values, import.destination)?;
    let eval_binding = register_bindings(bindings, values, eval)?;
    let implementation =
        implementations.register(GpuImplementation::ntt(false)).map_err(str::to_owned)?;
    let index =
        u32::try_from(operations.len()).map_err(|_| "too many GPU operations".to_owned())?;
    operations.push(CompiledGpuOp {
        implementation,
        arguments: Box::new([
            KernelArg::Value(import.destination),
            KernelArg::U32(0),
            KernelArg::Value(eval),
            KernelArg::U32(0),
            KernelArg::U32(source_binding),
            KernelArg::U32(eval_binding),
        ]),
        outputs: Box::new([eval]),
        device,
        grid: [1; 3],
        block: [1; 3],
        shared_bytes: 0,
        predecessors: Box::new([]),
        body: None,
    });
    let columns = values[eval.0 as usize]
        .ty
        .matrix_type()
        .ok_or("GPU imported evaluation is not a matrix")?
        .columns;
    producer.insert(eval, vec![(ColumnRange { start: 0, end: columns }, index)]);
    import.before_operation = index;
    templates.push(import);
    Ok(())
}

/// Stage artifact writes: one pre-allocated host slot per raw fragment and
/// member, a Graph copy into it, and a publish the I/O worker observes during
/// execution. Family members (`Some(index)`, in order) share one site per
/// fragment as consecutive occurrences.
#[allow(clippy::too_many_arguments)]
fn emit_artifact_export(
    ctx: &mut PhysicalLoweringContext<'_>,
    slots: &mut Vec<Arc<GpuExportSlot>>,
    export_templates: &mut Vec<ExportTemplate>,
    name: &str,
    members: &[(Option<usize>, PhysicalValueId)],
    artifact_type: ArtifactType,
    availability: ArtifactAvailability,
    public_export_source: Option<PhysicalValueId>,
) -> Result<(), String> {
    if members.is_empty() || (public_export_source.is_some() && members.len() != 1) {
        return Err("GPU artifact export needs members, and a trapdoor exactly one".into());
    }
    let mut staged = Vec::with_capacity(members.len());
    for &(index, export_source) in members {
        let physical = Arc::new(ctx.values[export_source.0 as usize].clone());
        let selected_parts = (0..physical.parts.len()).collect::<Vec<_>>();
        let mut export = PhysicalExport::from_parts(physical, &selected_parts)?;
        if let Some(public) = public_export_source {
            let public_physical = Arc::new(ctx.values[public.0 as usize].clone());
            let selected = (0..public_physical.parts.len()).collect::<Vec<_>>();
            export = export.with_trapdoor_public(public_physical, &selected)?;
        }
        let source_binding_base = if public_export_source.is_some() {
            register_all_parts(ctx.bindings, ctx.values, export_source)?
        } else if ctx.values[export_source.0 as usize].ty.matrix_type().is_some() {
            register_bindings(ctx.bindings, ctx.values, export_source)?
        } else {
            // A scalar leaf is one contiguous part.
            let binding = u32::try_from(ctx.bindings.len())
                .map_err(|_| "too many GPU graph bindings".to_owned())?;
            ctx.bindings.push(GpuBindingSource::PhysicalPart {
                value: export_source,
                part: 0,
                limb: 0,
            });
            binding
        };
        staged.push((index, export_source, Arc::new(export), source_binding_base));
    }
    let public_binding_base = public_export_source
        .map(|id| register_bindings(ctx.bindings, ctx.values, id))
        .transpose()?;
    let fragment_count = staged[0].2.fragments.len();
    if staged.iter().any(|(_, _, export, _)| export.fragments.len() != fragment_count) {
        return Err("GPU artifact family members have different raw layouts".into());
    }
    let copy =
        ctx.implementations.register(GpuImplementation::export_copy()).map_err(str::to_owned)?;
    let publish =
        ctx.implementations.register(GpuImplementation::export_publish()).map_err(str::to_owned)?;
    for fragment_index in 0..fragment_count {
        let site = u32::try_from(export_templates.len())
            .map_err(|_| "too many GPU artifact export sites".to_owned())?;
        for (index, export_source, export, source_binding_base) in &staged {
            let occurrence = index.unwrap_or(0);
            let fragment = &export.fragments[fragment_index];
            let slot = slots.len();
            let payload_bytes = usize::try_from(fragment.raw_bytes)
                .map_err(|_| "GPU artifact fragment exceeds host address space".to_owned())?;
            slots.push(Arc::new(
                GpuExportSlot::new(ctx.device, payload_bytes).map_err(|error| error.to_string())?,
            ));
            let payload_binding = u32::try_from(ctx.bindings.len())
                .map_err(|_| "too many GPU graph bindings".to_owned())?;
            ctx.bindings.push(GpuBindingSource::ExportSlotPayload { slot });
            let header_binding = u32::try_from(ctx.bindings.len())
                .map_err(|_| "too many GPU graph bindings".to_owned())?;
            ctx.bindings.push(GpuBindingSource::ExportSlotHeader { slot });
            let copy_index = u32::try_from(ctx.operations.len())
                .map_err(|_| "too many GPU operations".to_owned())?;
            let source_parts = ctx.values[export_source.0 as usize].parts.len();
            let (copy_source, part_index, binding_base) = match public_export_source {
                Some(public) if fragment_index >= source_parts => (
                    public,
                    u32::try_from(fragment_index - source_parts)
                        .map_err(|_| "GPU public export part index exceeds u32".to_owned())?,
                    public_binding_base
                        .ok_or_else(|| "GPU trapdoor public binding is missing".to_owned())?,
                ),
                _ => (
                    *export_source,
                    u32::try_from(fragment_index)
                        .map_err(|_| "GPU export fragment index exceeds u32".to_owned())?,
                    *source_binding_base,
                ),
            };
            let source_binding = binding_base
                .checked_add(part_index)
                .ok_or_else(|| "GPU export source binding overflows".to_owned())?;
            let slot_index =
                u32::try_from(slot).map_err(|_| "GPU export slot index exceeds u32".to_owned())?;
            ctx.operations.push(CompiledGpuOp {
                implementation: copy,
                arguments: Box::new([
                    KernelArg::Value(copy_source),
                    KernelArg::U32(part_index),
                    KernelArg::U32(slot_index),
                    KernelArg::U64(fragment.raw_bytes),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(payload_binding),
                ]),
                outputs: Box::new([]),
                device: ctx.device,
                grid: [0; 3],
                block: [0; 3],
                shared_bytes: 0,
                predecessors: all_predecessors(ctx.producer, copy_source),
                body: None,
            });
            let final_chunk = fragment_index + 1 == fragment_count;
            let occurrence = u64::try_from(occurrence)
                .map_err(|_| "GPU export occurrence exceeds u64".to_owned())?;
            ctx.operations.push(CompiledGpuOp {
                implementation: publish,
                arguments: Box::new([
                    KernelArg::U32(slot_index),
                    KernelArg::U64(occurrence),
                    KernelArg::U64(fragment.raw_offset),
                    KernelArg::U64(fragment.raw_bytes),
                    KernelArg::U32(site),
                    KernelArg::U32(u32::from(final_chunk)),
                    KernelArg::U32(header_binding),
                ]),
                outputs: Box::new([]),
                device: ctx.device,
                grid: [0; 3],
                block: [0; 3],
                shared_bytes: 0,
                predecessors: Box::new([copy_index]),
                body: None,
            });
            export_templates.push(ExportTemplate {
                name: name.to_owned(),
                index: *index,
                occurrence,
                site,
                slot,
                fragment_index,
                final_chunk,
                artifact_type: artifact_type.clone(),
                availability,
                export: Arc::clone(export),
            });
        }
    }
    Ok(())
}

/// Let full scratch matrices with disjoint lifetimes share allocations
/// (spec 7.3). A value group is every physical value viewing one allocation.
/// Its lifetime spans the top-level operations that reference it; an
/// operation with a nested body counts as one reference, and a value read
/// inside a replayed wave or external-I/O body stays live to that body's end.
/// An allocation is reused only when every operation that referenced its
/// previous occupant is already an ancestor of every writer of the new one,
/// so the frozen dependency structure, and with it the measured concurrency,
/// is unchanged. Protected values (inputs, outputs, wave-bound members,
/// imports, table candidates) and values read before any write keep their
/// own allocations.
fn share_scratch_allocations(
    values: &[PhysicalValue],
    owners: &mut BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    operations: &[CompiledGpuOp],
    replayed: &[(u32, u32)],
    protected: &BTreeSet<PhysicalValueId>,
) -> Result<(), String> {
    use crate::backend::BoundStorage;
    fn visit(op: &CompiledGpuOp, each: &mut dyn FnMut(PhysicalValueId, bool)) {
        for argument in op.arguments.iter() {
            if let KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) = argument {
                each(*id, false);
            }
        }
        for output in op.outputs.iter() {
            each(*output, true);
        }
        for inner in op.body.iter().flatten() {
            visit(inner, each);
        }
    }
    // Allocation of each full scratch matrix; any other value touching an
    // allocation makes it ineligible.
    let mut allocation_of = BTreeMap::<PhysicalValueId, *const ()>::new();
    let mut bound_of = BTreeMap::<*const (), BoundStorage>::new();
    let mut ineligible = BTreeSet::<*const ()>::new();
    for (id, owner) in owners.iter() {
        let physical = &values[id.0 as usize];
        let storages = owner.storages().collect::<Vec<_>>();
        let qualifies = physical.ty.matrix_type().is_some() &&
            matches!(
                physical.encodings.as_ref(),
                [PhysicalEncoding::FullCoeff] | [PhysicalEncoding::FullEval]
            ) &&
            storages.len() == 1 &&
            matches!(storages[0].0, StorageRef::Scratch(_)) &&
            !protected.contains(id);
        for (_, bound) in &storages {
            let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
            bound_of.entry(pointer).or_insert_with(|| (*bound).clone());
            if qualifies {
                allocation_of.insert(*id, pointer);
            } else {
                ineligible.insert(pointer);
            }
        }
    }
    struct Group {
        members: Vec<PhysicalValueId>,
        first_write: Option<usize>,
        first_reference: usize,
        last_reference: usize,
        references: BTreeSet<usize>,
        writers: BTreeSet<usize>,
    }
    let mut groups = BTreeMap::<*const (), Group>::new();
    for (&id, &pointer) in &allocation_of {
        if ineligible.contains(&pointer) {
            continue;
        }
        groups
            .entry(pointer)
            .or_insert_with(|| Group {
                members: Vec::new(),
                first_write: None,
                first_reference: usize::MAX,
                last_reference: 0,
                references: BTreeSet::new(),
                writers: BTreeSet::new(),
            })
            .members
            .push(id);
    }
    for (index, op) in operations.iter().enumerate() {
        visit(op, &mut |id, written| {
            let Some(group) = allocation_of.get(&id).and_then(|pointer| groups.get_mut(pointer))
            else {
                return;
            };
            group.first_reference = group.first_reference.min(index);
            group.last_reference = group.last_reference.max(index);
            group.references.insert(index);
            if written {
                group.writers.insert(index);
                group.first_write = Some(group.first_write.map_or(index, |first| first.min(index)));
            }
        });
    }
    for group in groups.values_mut() {
        for &(start, end) in replayed {
            let (start, end) = (start as usize, end as usize);
            if group.first_reference < start && group.references.range(start..end).next().is_some()
            {
                group.last_reference = group.last_reference.max(end - 1);
            }
        }
    }
    // The allocation's layout, independent of its storage slot.
    let layout = |group: &Group| {
        let mut physical = values[group.members[0].0 as usize].clone();
        for part in physical.parts.iter_mut() {
            part.storage = StorageRef::Scratch(0);
        }
        physical
    };
    let mut order = groups
        .iter()
        .filter(|(_, group)| group.first_write.is_some_and(|write| write <= group.first_reference))
        .map(|(pointer, group)| (group.first_write.unwrap_or(0), *pointer))
        .collect::<Vec<_>>();
    order.sort_unstable();
    // Does every op in `targets` precede `writer` through explicit edges?
    let precedes = |targets: &BTreeSet<usize>, writer: usize| {
        let lowest = *targets.first().unwrap_or(&writer);
        let mut pending = vec![writer];
        let mut seen = BTreeSet::new();
        let mut found = 0;
        while let Some(index) = pending.pop() {
            for &predecessor in operations[index].predecessors.iter() {
                let predecessor = predecessor as usize;
                if predecessor >= lowest && seen.insert(predecessor) {
                    found += usize::from(targets.contains(&predecessor));
                    pending.push(predecessor);
                }
            }
        }
        found == targets.len()
    };
    // Kept allocations per layout: (allocation, last reference, references).
    type Pool = Vec<(*const (), usize, BTreeSet<usize>)>;
    let mut pools = Vec::<(PhysicalValue, Pool)>::new();
    let mut replacements = BTreeMap::<*const (), *const ()>::new();
    for (first_write, pointer) in order {
        let group = &groups[&pointer];
        let key = layout(group);
        let pool = match pools.iter().position(|(existing, _)| *existing == key) {
            Some(index) => &mut pools[index].1,
            None => {
                pools.push((key, Vec::new()));
                &mut pools.last_mut().expect("pushed pool").1
            }
        };
        let reusable = pool.iter().position(|(_, last, references)| {
            *last < first_write && group.writers.iter().all(|&writer| precedes(references, writer))
        });
        match reusable {
            Some(slot) => {
                replacements.insert(pointer, pool[slot].0);
                pool[slot].1 = group.last_reference;
                pool[slot].2 = group.references.clone();
            }
            None => pool.push((pointer, group.last_reference, group.references.clone())),
        }
    }
    for (pointer, shared) in replacements {
        let replacement = std::collections::HashMap::from([(pointer, bound_of[&shared].clone())]);
        for id in &groups[&pointer].members {
            let owner = owners.get_mut(id).ok_or("GPU shared scratch owner is missing")?;
            if let Some(rebound) =
                owner.rebound(&replacement, owner.ready_events()).map_err(str::to_owned)?
            {
                *owner = Arc::new(rebound);
            }
        }
    }
    Ok(())
}

/// Allocate the physical frame during plan preparation. Every accepted node
/// has a direct physical operation or bounded control lowering; other
/// semantics fail preparation without a capture or host-compute fallback.
pub(crate) fn plan_physical_graph(
    backend: &GpuDcrtBackend,
    validated: &ValidatedGraph,
    logical: &FrozenGpuPlan,
    inputs: &BTreeMap<String, RuntimeValue>,
    integer_input_ranges: &BTreeMap<String, RangeInclusive<BigInt>>,
    artifact_payload_sizes: &BTreeMap<ArtifactKey, usize>,
) -> Result<PhysicalFrame, String> {
    let device = i32::try_from(
        *logical
            .contract
            .logical_to_physical_devices
            .first()
            .ok_or_else(|| "GPU plan has no physical device".to_owned())?,
    )
    .map_err(|_| "GPU device ID exceeds i32".to_owned())?;
    let scope = validated.source.root_scope();
    let checked = validated
        .scope(&mxx_ir_core::graph::FrozenGraphScopeId::Root)
        .ok_or_else(|| "validated root scope is missing".to_owned())?;
    let mut values = Vec::<PhysicalValue>::new();
    let mut real_owners = Vec::<Arc<GpuDeviceReal>>::new();
    let mut indexed_tables = Vec::<IndexedMatrixTableReplay>::new();
    let mut dynamic_export_resources =
        BTreeMap::<u32, (Arc<GpuDynamicExportTable>, Arc<GpuExportStatus>)>::new();
    let mut owners = BTreeMap::<PhysicalValueId, Arc<GpuResidentValue>>::new();
    let mut input_ids = BTreeMap::new();
    let mut integer_input_owners = BTreeMap::new();
    let mut real_input_owners = BTreeMap::new();
    let mut bytes_input_owners = BTreeMap::new();
    let mut output_ids = BTreeMap::new();
    let mut wire_ids = BTreeMap::<WireRef, PhysicalValueId>::new();
    let mut pending_imports = BTreeMap::<WireRef, (ImportTemplate, PhysicalValueId)>::new();
    let mut deferred_trapdoor_imports =
        Vec::<(WireRef, ArtifactKey, ManifestArtifact, ConcreteWireType)>::new();
    for (index, node) in scope.nodes().iter().enumerate() {
        let NodeKind::Input { name, artifact, .. } = node.kind() else { continue };
        let node_id =
            u64::try_from(index).map_err(|_| "GPU graph has too many nodes".to_owned())?;
        let wire = WireRef { node: mxx_ir_core::types::NodeId(node_id), port: Port(0) };
        if let Some(artifact) = artifact {
            let used = scope.nodes().iter().any(|consumer| {
                scope.arguments(consumer).is_some_and(|arguments| arguments.contains(&wire))
            }) || validated.source.outputs().values().any(|output| output.value == wire);
            if !used {
                continue;
            }
            let descriptor = checked
                .artifact_inputs
                .get(&wire)
                .ok_or_else(|| "GPU artifact import lacks validated descriptor".to_owned())?
                .clone();
            if descriptor.family_count.is_some() {
                // A family is only a descriptor here. Parallel Zip lowering
                // requests individual members as its waves reach them.
                continue;
            }
            if let Some(ty @ ConcreteWireType::Trapdoor { .. }) = checked.wire_types.get(&wire) {
                deferred_trapdoor_imports.push((
                    wire,
                    ArtifactKey {
                        production: artifact.production_id.clone(),
                        name: artifact.artifact_name.clone(),
                        index: None,
                    },
                    descriptor,
                    ty.clone(),
                ));
                continue;
            }
            if let Some(ConcreteWireType::Bytes { length }) = checked.wire_types.get(&wire) {
                if *length == 0 {
                    return Err("GPU zero-length artifact bytes need an empty physical leaf".into());
                }
                let params = backend.control_parameters_on_device(device)?;
                let native = Arc::new(
                    GpuDeviceBytes::new(&params, device, *length)
                        .map_err(|error| error.to_string())?,
                );
                let storage = StorageRef::Input(
                    u32::try_from(values.len()).map_err(|_| "too many GPU import storages")?,
                );
                let physical = PhysicalValue {
                    ty: ConcreteWireType::Bytes { length: *length },
                    encodings: Box::new([PhysicalEncoding::Bytes]),
                    parts: Box::new([PhysicalPart {
                        leaf: 0,
                        storage,
                        device,
                        view: PhysicalView {
                            byte_offset: 0,
                            origin: Box::new([0]),
                            extent: Box::new([*length as u64]),
                            byte_strides: Box::new([1]),
                            element_bytes: 1,
                        },
                    }]),
                    integer_ranges: BTreeMap::new(),
                };
                let resident = GpuResidentValue::new(
                    Arc::new(physical.clone()),
                    BTreeMap::from([(
                        storage,
                        BoundStorage::from_device_bytes(Arc::clone(&native))?,
                    )]),
                    Box::new([]),
                )
                .map_err(str::to_owned)?;
                let destination = value_id(values.len())?;
                values.push(physical);
                owners.insert(destination, Arc::new(resident));
                wire_ids.insert(wire, destination);
                pending_imports.insert(
                    wire,
                    (
                        ImportTemplate {
                            before_operation: 0,
                            key: ArtifactKey {
                                production: artifact.production_id.clone(),
                                name: artifact.artifact_name.clone(),
                                index: None,
                            },
                            descriptor,
                            expected_type: ArtifactType::Bytes { length: *length },
                            staged: false,
                            destination,
                            upload_owner: ImportDestination::Bytes {
                                owner: native,
                                length: *length,
                            },
                        },
                        destination,
                    ),
                );
                continue;
            }
            if let Some(ty @ ConcreteWireType::TypedBlob { .. }) = checked.wire_types.get(&wire) {
                let key = ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: None,
                };
                let payload_size = *artifact_payload_sizes.get(&key).ok_or_else(|| {
                    "GPU TypedBlob artifact requires plan_with_store for payload size".to_owned()
                })?;
                let capacity = payload_size
                    .checked_add(8)
                    .ok_or("GPU TypedBlob physical capacity overflows")?;
                let params = backend.control_parameters_on_device(device)?;
                let native = Arc::new(
                    GpuDeviceBytes::new(&params, device, capacity)
                        .map_err(|error| error.to_string())?,
                );
                let storage = StorageRef::Input(
                    u32::try_from(values.len()).map_err(|_| "too many GPU import storages")?,
                );
                let physical = PhysicalValue {
                    ty: ty.clone(),
                    encodings: Box::new([PhysicalEncoding::TypedBlobLengthPrefixed]),
                    parts: Box::new([PhysicalPart {
                        leaf: 0,
                        storage,
                        device,
                        view: PhysicalView {
                            byte_offset: 0,
                            origin: Box::new([0]),
                            extent: Box::new([capacity as u64]),
                            byte_strides: Box::new([1]),
                            element_bytes: 1,
                        },
                    }]),
                    integer_ranges: BTreeMap::new(),
                };
                let resident = GpuResidentValue::new(
                    Arc::new(physical.clone()),
                    BTreeMap::from([(
                        storage,
                        BoundStorage::from_device_bytes(Arc::clone(&native))?,
                    )]),
                    Box::new([]),
                )
                .map_err(str::to_owned)?;
                let destination = value_id(values.len())?;
                values.push(physical);
                owners.insert(destination, Arc::new(resident));
                wire_ids.insert(wire, destination);
                pending_imports.insert(
                    wire,
                    (
                        ImportTemplate {
                            before_operation: 0,
                            key,
                            descriptor,
                            expected_type: ArtifactType::from_wire_type(ty)
                                .ok_or("GPU typed blob import has no artifact type")?,
                            staged: false,
                            destination,
                            upload_owner: ImportDestination::Bytes {
                                owner: native,
                                length: capacity,
                            },
                        },
                        destination,
                    ),
                );
                continue;
            }
            if let Some(
                ty @ (ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. }),
            ) = checked.wire_types.get(&wire)
            {
                let storage = StorageRef::Input(
                    u32::try_from(values.len())
                        .map_err(|_| "too many GPU bounded import storages")?,
                );
                let (physical, resident, native, _) =
                    compact_value_owner(backend, device, ty.clone(), storage)?;
                let destination = value_id(values.len())?;
                values.push(physical);
                owners.insert(destination, resident);
                wire_ids.insert(wire, destination);
                pending_imports.insert(
                    wire,
                    (
                        ImportTemplate {
                            before_operation: 0,
                            key: ArtifactKey {
                                production: artifact.production_id.clone(),
                                name: artifact.artifact_name.clone(),
                                index: None,
                            },
                            descriptor,
                            expected_type: ArtifactType::from_wire_type(ty)
                                .ok_or("GPU bounded import has no artifact type")?,
                            staged: false,
                            destination,
                            upload_owner: ImportDestination::Bounded {
                                owner: native,
                                ty: ty.clone(),
                            },
                        },
                        destination,
                    ),
                );
                continue;
            }
            if let Some(ty @ ConcreteWireType::Int) = checked.wire_types.get(&wire) {
                let key = ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: None,
                };
                let payload_size = *artifact_payload_sizes.get(&key).ok_or_else(|| {
                    "GPU Int artifact requires plan_with_store for payload size".to_owned()
                })?;
                if payload_size == 0 {
                    return Err("GPU Int artifact has no canonical signed byte capacity".into());
                }
                let bits = payload_size.checked_mul(8).ok_or("GPU Int artifact width overflows")?;
                let limit = BigInt::from(1u8) << (bits - 1);
                let range = -&limit..=limit - 1;
                let storage = StorageRef::Input(
                    u32::try_from(values.len()).map_err(|_| "too many GPU Int import storages")?,
                );
                let (physical, resident, native) = planned_integer_input(
                    backend,
                    device,
                    ty,
                    &[BigInt::from(0u8)],
                    &range,
                    storage,
                )?;
                let destination = value_id(values.len())?;
                values.push(physical);
                owners.insert(destination, resident);
                wire_ids.insert(wire, destination);
                pending_imports.insert(
                    wire,
                    (
                        ImportTemplate {
                            before_operation: 0,
                            key,
                            descriptor,
                            expected_type: ArtifactType::Int,
                            staged: false,
                            destination,
                            upload_owner: ImportDestination::Signed {
                                owner: native,
                                ty: ty.clone(),
                            },
                        },
                        destination,
                    ),
                );
                continue;
            }
            let ty = checked
                .wire_types
                .get(&wire)
                .and_then(ConcreteWireType::matrix_type)
                .ok_or_else(|| "GPU artifact import currently needs one matrix".to_owned())?
                .clone();
            let native =
                backend.allocate_physical_matrix(&ty, device, PhysicalEncoding::FullCoeff)?;
            let storage = StorageRef::Input(
                u32::try_from(values.len())
                    .map_err(|_| "too many GPU import storages".to_owned())?,
            );
            let (physical, owner) =
                physical_matrix(&ty, PhysicalEncoding::FullCoeff, storage, Arc::clone(&native))?;
            let coefficient = value_id(values.len())?;
            values.push(physical);
            owners.insert(coefficient, owner);
            let eval_native =
                backend.allocate_physical_matrix(&ty, device, PhysicalEncoding::FullEval)?;
            let eval_storage = StorageRef::Scratch(
                u32::try_from(values.len())
                    .map_err(|_| "too many GPU import stagings".to_owned())?,
            );
            let (physical, owner) =
                physical_matrix(&ty, PhysicalEncoding::FullEval, eval_storage, eval_native)?;
            let eval = value_id(values.len())?;
            values.push(physical);
            owners.insert(eval, owner);
            wire_ids.insert(wire, eval);
            pending_imports.insert(
                wire,
                (
                    ImportTemplate {
                        before_operation: 0,
                        key: ArtifactKey {
                            production: artifact.production_id.clone(),
                            name: artifact.artifact_name.clone(),
                            index: None,
                        },
                        descriptor,
                        expected_type: ArtifactType::from_wire_type(
                            checked.wire_types.get(&wire).ok_or("GPU import has no wire type")?,
                        )
                        .ok_or("GPU import has no artifact type")?,
                        staged: false,
                        destination: coefficient,
                        upload_owner: ImportDestination::Matrix { owner: native, ty },
                    },
                    eval,
                ),
            );
            continue;
        }
        let expected = checked
            .wire_types
            .get(&wire)
            .ok_or_else(|| format!("GPU input {name} has no validated type"))?;
        let supplied = inputs.get(name).ok_or_else(|| format!("missing GPU input {name}"))?;
        if let Some(integers) = host_integer_values(supplied, expected)? {
            let range = integer_input_ranges
                .get(name)
                .ok_or_else(|| format!("GPU host integer input {name} has no declared range"))?;
            let storage = StorageRef::Input(
                u32::try_from(values.len())
                    .map_err(|_| "too many GPU integer input storages".to_owned())?,
            );
            let (physical, resident, native) =
                planned_integer_input(backend, device, expected, &integers, range, storage)?;
            let id = value_id(values.len())?;
            values.push(physical);
            owners.insert(id, resident);
            integer_input_owners.insert(name.clone(), native);
            input_ids.insert(name.clone(), id);
            wire_ids.insert(wire, id);
            continue;
        }
        if let RuntimeValue::Real(real) = supplied {
            let storage = StorageRef::Input(
                u32::try_from(values.len())
                    .map_err(|_| "too many GPU real input storages".to_owned())?,
            );
            let (physical, resident, native) =
                planned_real_input(backend, device, expected, *real, storage)?;
            let id = value_id(values.len())?;
            values.push(physical);
            owners.insert(id, resident);
            real_input_owners.insert(name.clone(), Arc::clone(&native));
            real_owners.push(native);
            input_ids.insert(name.clone(), id);
            wire_ids.insert(wire, id);
            continue;
        }
        if let (ConcreteWireType::Bytes { length: 32 }, RuntimeValue::Bytes(bytes)) =
            (expected, supplied)
        {
            let key: [u8; 32] = bytes
                .as_ref()
                .try_into()
                .map_err(|_| format!("GPU bytes input {name} is not 32 bytes"))?;
            let params = backend.control_parameters_on_device(device)?;
            let native =
                Arc::new(GpuDeviceSeed::new(&params, device).map_err(|error| error.to_string())?);
            native.upload(&key).map_err(|error| error.to_string())?;
            let storage = StorageRef::Input(
                u32::try_from(values.len())
                    .map_err(|_| "too many GPU bytes input storages".to_owned())?,
            );
            let physical = PhysicalValue {
                ty: expected.clone(),
                encodings: Box::new([PhysicalEncoding::Bytes]),
                parts: Box::new([PhysicalPart {
                    leaf: 0,
                    storage,
                    device,
                    view: PhysicalView {
                        byte_offset: 0,
                        origin: Box::new([0]),
                        extent: Box::new([32]),
                        byte_strides: Box::new([1]),
                        element_bytes: 1,
                    },
                }]),
                integer_ranges: BTreeMap::new(),
            };
            let resident = GpuResidentValue::new(
                Arc::new(physical.clone()),
                BTreeMap::from([(storage, BoundStorage::from_device_seed(Arc::clone(&native))?)]),
                Box::new([]),
            )
            .map_err(str::to_owned)?;
            let id = value_id(values.len())?;
            values.push(physical);
            owners.insert(id, Arc::new(resident));
            bytes_input_owners.insert(name.clone(), native);
            input_ids.insert(name.clone(), id);
            wire_ids.insert(wire, id);
            continue;
        }
        let resident = resident_input_with_type(supplied, expected, device)?;
        let resident =
            with_declared_integer_range(resident, expected, integer_input_ranges.get(name))?;
        if resident.wire_type() != expected ||
            resident.physical().parts.iter().any(|part| part.device != device)
        {
            return Err(format!("GPU input {name} has the wrong type or device"));
        }
        let id = value_id(values.len())?;
        values.push(resident.physical().as_ref().clone());
        owners.insert(id, resident);
        input_ids.insert(name.clone(), id);
        wire_ids.insert(wire, id);
    }
    if input_ids.len() != inputs.len() {
        return Err("GPU input set includes an undeclared value".into());
    }
    let mut implementations = GpuImplementationRegistry::default();
    let mut operations = Vec::<CompiledGpuOp>::new();
    let mut bindings = Vec::<GpuBindingSource>::new();
    let mut producer = BTreeMap::<PhysicalValueId, Vec<(ColumnRange, u32)>>::new();
    let mut family_member_producers =
        BTreeMap::<(PhysicalValueId, usize), Vec<(ColumnRange, u32)>>::new();
    let mut slots = Vec::<Arc<GpuExportSlot>>::new();
    let mut export_templates = Vec::<ExportTemplate>::new();
    let mut coefficient_exports = BTreeMap::<PhysicalValueId, PhysicalValueId>::new();
    let mut control_resets = Vec::<ControlReset>::new();
    let mut sample_seeds = Vec::<SampleSeed>::new();
    let mut hash_resources = BTreeMap::<u32, GpuHashResourceSpec>::new();
    let real_output_owners = BTreeMap::new();
    let mut preimage_replays = Vec::<PreimageReplay>::new();
    let mut trapdoor_public_ids = BTreeMap::new();
    let mut waves = Vec::<PhysicalWave>::new();
    let mut import_templates = Vec::<ImportTemplate>::new();
    let mut external_io_loops = Vec::<ExternalIoLoop>::new();
    let mut external_io_imports = Vec::<ExternalIoImport>::new();
    let mut crt_resource_next = 0u32;
    let mut converted = BTreeMap::new();
    let mut integer_status = BTreeMap::new();
    // Every root node lowers against the same plan tables.
    macro_rules! root_context {
        () => {
            PhysicalLoweringContext {
                validated,
                integer_input_ranges,
                artifact_payload_sizes,
                backend,
                logical,
                device,
                values: &mut values,
                owners: &mut owners,
                wire_ids: &mut wire_ids,
                implementations: &mut implementations,
                operations: &mut operations,
                bindings: &mut bindings,
                producer: &mut producer,
                family_member_producers: &mut family_member_producers,
                control_resets: &mut control_resets,
                sample_seeds: &mut sample_seeds,
                hash_resources: &mut hash_resources,
                real_owners: &mut real_owners,
                indexed_tables: &mut indexed_tables,
                dynamic_export_resources: &mut dynamic_export_resources,
                device_loop_indices: BTreeMap::new(),
                lanes: 1,
                active_parallel_template: None,
                active_parallel_instances: Vec::new(),
                device_body: false,
                preimage_replays: &mut preimage_replays,
                trapdoor_public_ids: &mut trapdoor_public_ids,
                waves: &mut waves,
                import_templates: &mut import_templates,
                external_io_loops: &mut external_io_loops,
                external_io_imports: &mut external_io_imports,
                crt_resource_next: &mut crt_resource_next,
                converted: &mut converted,
                integer_status: &mut integer_status,
            }
        };
    }
    for (wire, key, descriptor, ty) in deferred_trapdoor_imports {
        let mut ctx = root_context!();
        let (destination, graph_value, upload_owner, before_operation) =
            allocate_typed_import_destination(&mut ctx, &ty, &key, None)?;
        ctx.wire_ids.insert(wire, graph_value);
        pending_imports.insert(
            wire,
            (
                ImportTemplate {
                    before_operation,
                    key,
                    descriptor,
                    expected_type: ArtifactType::from_wire_type(&ty)
                        .ok_or("GPU trapdoor import has no artifact type")?,
                    staged: false,
                    destination,
                    upload_owner,
                },
                graph_value,
            ),
        );
    }
    for (index, node) in scope.nodes().iter().enumerate() {
        if matches!(node.kind(), NodeKind::Input { .. }) {
            continue;
        }
        let node_id = mxx_ir_core::types::NodeId(
            u64::try_from(index).map_err(|_| "GPU graph has too many nodes".to_owned())?,
        );
        let arguments = scope
            .arguments(node)
            .ok_or_else(|| "GPU root node has no validated arguments".to_owned())?;
        let consumes_artifact = arguments.iter().any(|wire| pending_imports.contains_key(wire));
        if consumes_artifact {
            let active_arguments = if let NodeKind::SequentialLoop(spec) = node.kind() {
                if spec.carried_count > arguments.len() {
                    return Err("GPU sequential carried count exceeds its arguments".into());
                }
                let count = finite_loop_count(
                    &FrozenGraphScopeId::Root,
                    node_id,
                    node.kind(),
                    &validated.bindings,
                )?;
                if count > 0 {
                    let child_id = validated
                        .source
                        .child_scope_id(&FrozenGraphScopeId::Root, node_id)
                        .ok_or_else(|| {
                            "GPU sequential artifact consumer has no child".to_owned()
                        })?;
                    let child = validated
                        .source
                        .scope(&child_id)
                        .ok_or_else(|| "GPU sequential artifact child is absent".to_owned())?;
                    for (position, wire) in arguments.iter().enumerate().skip(spec.carried_count) {
                        if pending_imports.contains_key(wire) &&
                            !matches!(
                                child
                                    .inputs()
                                    .get(position)
                                    .and_then(|input| child.node(input.node))
                                    .map(mxx_ir_core::graph::NodeHandle::kind),
                                Some(NodeKind::Input { artifact: Some(_), .. })
                            )
                        {
                            return Err("GPU sequential artifact body input has no scoped import descriptor".into());
                        }
                    }
                }
                arguments[..spec.carried_count].to_vec()
            } else if matches!(
                node.kind(),
                NodeKind::SubgraphCall(_) |
                    NodeKind::MatrixBinary(_) |
                    NodeKind::MatrixNegate |
                    NodeKind::MatrixMulAccumulate { .. } |
                    NodeKind::MatrixMulSmallRhs |
                    NodeKind::MatrixScale { .. } |
                    NodeKind::RingAutomorphism { .. } |
                    NodeKind::ModulusSwitch { .. } |
                    NodeKind::GadgetDecompose { .. } |
                    NodeKind::ExtractCoefficient { .. } |
                    NodeKind::LiftIntegerToConstantPolynomial { .. } |
                    NodeKind::ThresholdDecode { .. } |
                    NodeKind::PackPolynomialCoefficients { .. } |
                    NodeKind::PolynomialValues { .. } |
                    NodeKind::Transpose |
                    NodeKind::Tensor |
                    NodeKind::Concat { .. } |
                    NodeKind::Slice { .. } |
                    NodeKind::ModulusReduce { .. } |
                    NodeKind::CenteredRoundDivide { .. } |
                    NodeKind::RnsModUp { .. } |
                    NodeKind::RnsModDown { .. } |
                    NodeKind::BlockModSwitch { .. } |
                    NodeKind::CrtRecompose { .. } |
                    NodeKind::CenteredRebase { .. } |
                    NodeKind::HashSample { .. }
            ) {
                arguments.to_vec()
            } else {
                return Err(
                    "GPU artifact first consumer needs a planned conditional import boundary"
                        .into(),
                );
            };
            for wire in active_arguments {
                activate_matrix_import(
                    wire,
                    &mut pending_imports,
                    &values,
                    &mut implementations,
                    &mut operations,
                    &mut bindings,
                    &mut producer,
                    &mut import_templates,
                    device,
                )?;
            }
        }
        let mut ctx = root_context!();
        if matches!(node.kind(), NodeKind::MatrixBinary(_)) {
            let choices = logical
                .nodes
                .iter()
                .filter(|choice| {
                    choice.key.site == node_id.0 &&
                        choice.key.shape_class == 0 &&
                        choice.key.instance_class == 0 &&
                        choice.loop_site.is_none()
                })
                .collect::<Vec<_>>();
            let [choice] = choices.as_slice() else {
                return Err(format!("GPU node {:?} has no unique frozen root choice", node_id));
            };
            lower_matrix_node(&mut ctx, scope, node_id, node, &checked.wire_types, choice)?;
        } else if matches!(node.kind(), NodeKind::ConstantMatrix { .. }) {
            lower_static_matrix_node(
                &mut ctx,
                node_id,
                node,
                &validated.bindings,
                &checked.wire_types,
            )?;
        } else if matches!(
            node.kind(),
            NodeKind::UniformResidueSample { .. } |
                NodeKind::UniformIntervalSample { .. } |
                NodeKind::GaussianSample { .. }
        ) {
            lower_sample_matrix_node(
                &mut ctx,
                &FrozenGraphScopeId::Root,
                node_id,
                node,
                &validated.bindings,
                &checked.wire_types,
            )?;
        } else if matches!(node.kind(), NodeKind::TrapdoorSample { .. }) {
            lower_trapdoor_sample_node(
                &mut ctx,
                &FrozenGraphScopeId::Root,
                node_id,
                node,
                &validated.bindings,
                &checked.wire_types,
            )?;
        } else if matches!(node.kind(), NodeKind::HashSample { .. }) {
            lower_hash_sample_node(
                &mut ctx,
                &FrozenGraphScopeId::Root,
                node_id,
                node,
                &validated.bindings,
                &checked.wire_types,
            )?;
        } else if matches!(node.kind(), NodeKind::GadgetTrapdoor { .. }) {
            lower_gadget_trapdoor_node(
                &mut ctx,
                node_id,
                node,
                &validated.bindings,
                &checked.wire_types,
            )?;
        } else if matches!(node.kind(), NodeKind::PreimageSample { .. }) {
            lower_preimage_sample_node(
                &mut ctx,
                &FrozenGraphScopeId::Root,
                node_id,
                node,
                &validated.bindings,
                &checked.wire_types,
            )?;
        } else if matches!(
            node.kind(),
            NodeKind::RnsModUp { .. } |
                NodeKind::RnsModDown { .. } |
                NodeKind::BlockModSwitch { .. } |
                NodeKind::CrtRecompose { .. }
        ) {
            if matches!(node.kind(), NodeKind::CrtRecompose { .. }) {
                lower_crt_recompose_node(
                    &mut ctx,
                    scope,
                    node_id,
                    node,
                    &validated.bindings,
                    &checked.wire_types,
                )?;
            } else {
                lower_rns_conversion_node(
                    &mut ctx,
                    scope,
                    node_id,
                    node,
                    &validated.bindings,
                    &checked.wire_types,
                )?;
            }
        } else if matches!(node.kind(), NodeKind::CenteredRebase { .. }) {
            lower_centered_rebase_node(&mut ctx, scope, node_id, node, &checked.wire_types)?;
        } else {
            lower_control_node(
                &mut ctx,
                &validated.source,
                &mxx_ir_core::graph::FrozenGraphScopeId::Root,
                node_id,
                node,
                &validated.bindings,
            )?;
        }
    }
    for (name, output_root) in validated.source.outputs() {
        activate_matrix_import(
            output_root.value,
            &mut pending_imports,
            &values,
            &mut implementations,
            &mut operations,
            &mut bindings,
            &mut producer,
            &mut import_templates,
            device,
        )?;
        let source = *wire_ids
            .get(&output_root.value)
            .ok_or_else(|| format!("GPU output {name} has no physical value"))?;
        let scalar_type = values[source.0 as usize].ty.clone();
        if matches!(&scalar_type, ConcreteWireType::IndexedFamily { count: 0, .. }) &&
            output_root.availability.is_none()
        {
            // The typed empty family has no device parts and no replay work.
            output_ids.insert(name.clone(), source);
            continue;
        }
        let integer_family = matches!(&scalar_type, ConcreteWireType::IndexedFamily { element, .. }
            if element.as_ref() == &ConcreteWireType::Int);
        if matches!(
            scalar_type,
            ConcreteWireType::Int |
                ConcreteWireType::Bool |
                ConcreteWireType::ConstantInt |
                ConcreteWireType::ConstantBool
        ) || integer_family
        {
            if let Some(availability) = output_root.availability {
                // Each family member is its own indexed Int artifact.
                let members = match &scalar_type {
                    ConcreteWireType::IndexedFamily { count, .. } => {
                        (0..*count).map(Some).collect()
                    }
                    ConcreteWireType::Int | ConcreteWireType::ConstantInt => vec![None],
                    _ => return Err(format!("GPU output {name} is not artifact-compatible")),
                };
                let source_owner = Arc::clone(&owners[&source]);
                let mut ctx = root_context!();
                let mut exported = Vec::with_capacity(members.len());
                for index in members {
                    let export_source = match index {
                        None => source,
                        Some(index) => {
                            let member = static_family_member(&source_owner, index)?;
                            let id = value_id(ctx.values.len())?;
                            ctx.values.push(member.physical().as_ref().clone());
                            ctx.owners.insert(id, member);
                            if let Some(writers) = ctx.producer.get(&source).cloned() {
                                ctx.producer.insert(id, writers);
                            }
                            id
                        }
                    };
                    exported.push((index, export_source));
                }
                emit_artifact_export(
                    &mut ctx,
                    &mut slots,
                    &mut export_templates,
                    name,
                    &exported,
                    ArtifactType::Int,
                    availability,
                    None,
                )?;
            }
            let range =
                if matches!(scalar_type, ConcreteWireType::Bool | ConcreteWireType::ConstantBool) {
                    BigInt::from(0)..=BigInt::from(1)
                } else {
                    values[source.0 as usize]
                        .integer_ranges
                        .get(&0)
                        .cloned()
                        .ok_or_else(|| format!("GPU integer output {name} has no proven range"))?
                };
            // A returned family's layout follows its range, not its producer,
            // so outputs of different graphs rebind to one another's plans.
            let canonical =
                range.start().sign() != num_bigint::Sign::Minus && range.end().bits() <= 64;
            let mut ctx = root_context!();
            let returned = if integer_family {
                allocate_return_integer_family_value(&mut ctx, scalar_type, range, canonical)?
            } else {
                allocate_return_integer_value(&mut ctx, scalar_type, range)?
            };
            emit_integer_operation(
                &mut ctx,
                GpuIntegerOperation::Copy,
                returned,
                source,
                None,
                None,
                0,
            )?;
            output_ids.insert(name.clone(), returned);
            continue;
        }
        if matches!(values[source.0 as usize].ty, ConcreteWireType::IndexedFamily { .. }) {
            if let Some(availability) = output_root.availability {
                let mut ctx = root_context!();
                lower_wave_family_artifact_export(
                    &mut ctx,
                    name,
                    source,
                    availability,
                    &mut slots,
                    &mut export_templates,
                )?;
            }
            output_ids.insert(name.clone(), source);
            continue;
        }
        if matches!(
            values[source.0 as usize].ty,
            ConcreteWireType::Preimage { .. } | ConcreteWireType::SmallMatrix { .. }
        ) && output_root.availability.is_none()
        {
            let source_value = &values[source.0 as usize];
            let [source_part] = source_value.parts.as_ref() else {
                return Err("GPU bounded return needs one compact physical part".into());
            };
            if matches!(source_part.storage, StorageRef::Output(_)) {
                output_ids.insert(name.clone(), source);
                continue;
            }
            if !matches!(
                source_value.encodings.as_ref(),
                [PhysicalEncoding::CompactCoeff { .. }] |
                    [PhysicalEncoding::CompactCoeffPerCrtLimb { .. }]
            ) || source_part.view.origin.iter().any(|origin| *origin != 0)
            {
                return Err("GPU bounded return source is not a complete compact owner".into());
            }
            let storage = StorageRef::Output(
                u32::try_from(values.len()).map_err(|_| "too many GPU bounded returns")?,
            );
            let (physical, resident, _, _) =
                compact_value_owner(backend, device, source_value.ty.clone(), storage)?;
            if physical.parts[0].view.extent != source_part.view.extent ||
                physical.encodings != source_value.encodings
            {
                return Err("GPU bounded return owner changed its physical shape".into());
            }
            let bytes = physical.parts[0]
                .view
                .extent
                .iter()
                .try_fold(1u64, |size, extent| size.checked_mul(*extent))
                .ok_or("GPU bounded return copy size overflows")?;
            let returned = value_id(values.len())?;
            values.push(physical);
            owners.insert(returned, resident);
            let source_binding =
                u32::try_from(bindings.len()).map_err(|_| "too many GPU bounded bindings")?;
            bindings.push(GpuBindingSource::PhysicalPart { value: source, part: 0, limb: 0 });
            let return_binding =
                u32::try_from(bindings.len()).map_err(|_| "too many GPU bounded bindings")?;
            bindings.push(GpuBindingSource::PhysicalPart { value: returned, part: 0, limb: 0 });
            let implementation =
                implementations.register(GpuImplementation::copy()).map_err(str::to_owned)?;
            let index = u32::try_from(operations.len()).map_err(|_| "too many GPU operations")?;
            operations.push(CompiledGpuOp {
                implementation,
                arguments: Box::new([
                    KernelArg::Value(source),
                    KernelArg::U32(0),
                    KernelArg::Value(returned),
                    KernelArg::U32(0),
                    KernelArg::U64(bytes),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(return_binding),
                ]),
                outputs: Box::new([returned]),
                device,
                grid: [1; 3],
                block: [1; 3],
                shared_bytes: 0,
                predecessors: all_predecessors(&producer, source),
                body: None,
            });
            producer.insert(returned, vec![(ColumnRange { start: 0, end: 1 }, index)]);
            output_ids.insert(name.clone(), returned);
            continue;
        }
        if matches!(
            values[source.0 as usize].ty,
            ConcreteWireType::Bytes { .. } | ConcreteWireType::TypedBlob { .. }
        ) && output_root.availability.is_none()
        {
            let source_value = &values[source.0 as usize];
            let [source_part] = source_value.parts.as_ref() else {
                return Err("GPU bytes return source needs one physical part".into());
            };
            if !matches!(
                source_value.encodings.as_ref(),
                [PhysicalEncoding::Bytes] | [PhysicalEncoding::TypedBlobLengthPrefixed]
            ) || source_part.view.origin.as_ref() != [0] ||
                source_part.view.byte_strides.as_ref() != [1]
            {
                return Err("GPU bytes return source is not contiguous".into());
            }
            let capacity = usize::try_from(source_part.view.extent[0])
                .map_err(|_| "GPU bytes return capacity exceeds usize")?;
            let params = backend.control_parameters_on_device(device)?;
            let native = Arc::new(
                GpuDeviceBytes::new(&params, device, capacity)
                    .map_err(|error| error.to_string())?,
            );
            native.wait_until_ready().map_err(|error| error.to_string())?;
            let storage = StorageRef::Output(
                u32::try_from(values.len()).map_err(|_| "too many GPU bytes return storages")?,
            );
            let mut physical = source_value.clone();
            physical.parts[0].storage = storage;
            let resident = GpuResidentValue::new(
                Arc::new(physical.clone()),
                BTreeMap::from([(storage, BoundStorage::from_device_bytes(native)?)]),
                Box::new([]),
            )
            .map_err(str::to_owned)?;
            let returned = value_id(values.len())?;
            values.push(physical);
            owners.insert(returned, Arc::new(resident));
            let source_binding =
                u32::try_from(bindings.len()).map_err(|_| "too many GPU bytes bindings")?;
            bindings.push(GpuBindingSource::PhysicalPart { value: source, part: 0, limb: 0 });
            let return_binding =
                u32::try_from(bindings.len()).map_err(|_| "too many GPU bytes bindings")?;
            bindings.push(GpuBindingSource::PhysicalPart { value: returned, part: 0, limb: 0 });
            let implementation =
                implementations.register(GpuImplementation::copy()).map_err(str::to_owned)?;
            let index = u32::try_from(operations.len()).map_err(|_| "too many GPU operations")?;
            operations.push(CompiledGpuOp {
                implementation,
                arguments: Box::new([
                    KernelArg::Value(source),
                    KernelArg::U32(0),
                    KernelArg::Value(returned),
                    KernelArg::U32(0),
                    KernelArg::U64(capacity as u64),
                    KernelArg::U32(source_binding),
                    KernelArg::U32(return_binding),
                ]),
                outputs: Box::new([returned]),
                device,
                grid: [1; 3],
                block: [1; 3],
                shared_bytes: 0,
                predecessors: all_predecessors(&producer, source),
                body: None,
            });
            producer.insert(returned, vec![(ColumnRange { start: 0, end: 1 }, index)]);
            output_ids.insert(name.clone(), returned);
            continue;
        }
        if matches!(scalar_type, ConcreteWireType::Real | ConcreteWireType::ConstantReal) {
            if output_root.availability.is_some() {
                return Err("GPU real artifact output needs a direct scalar export site".into());
            }
            let mut ctx = root_context!();
            let returned = allocate_real_value(&mut ctx, scalar_type)?;
            emit_real_operation(
                &mut ctx,
                GpuRealOperation::Copy,
                returned,
                Some(source),
                None,
                0.0,
            )?;
            output_ids.insert(name.clone(), returned);
            continue;
        }
        if values[source.0 as usize].encodings.as_ref() == [PhysicalEncoding::PublicGadgetEval] &&
            output_root.availability.is_none()
        {
            let public = *trapdoor_public_ids
                .get(&source)
                .ok_or_else(|| "GPU public gadget return has no paired public matrix".to_owned())?;
            let trapdoor_ty = values[source.0 as usize].ty.clone();
            let ConcreteWireType::Trapdoor { matrix, .. } = &trapdoor_ty else {
                return Err("GPU public gadget return is not trapdoor typed".into());
            };
            let native =
                backend.allocate_physical_matrix(matrix, device, PhysicalEncoding::FullEval)?;
            let storage = StorageRef::Output(
                u32::try_from(values.len())
                    .map_err(|_| "too many GPU public gadget return storages".to_owned())?,
            );
            let (mut physical, owner) =
                physical_matrix(matrix, PhysicalEncoding::FullEval, storage, native)?;
            physical.ty = trapdoor_ty;
            physical.encodings = Box::new([PhysicalEncoding::PublicGadgetEval]);
            let resident =
                owner.with_physical_view(Arc::new(physical.clone())).map_err(str::to_owned)?;
            let returned = value_id(values.len())?;
            values.push(physical);
            owners.insert(returned, Arc::new(resident));
            let mut ctx = root_context!();
            emit_matrix_operation(
                &mut ctx,
                GpuImplementation::matrix_copy_view(),
                &[public],
                returned,
            )?;
            ctx.trapdoor_public_ids.insert(returned, public);
            output_ids.insert(name.clone(), returned);
            continue;
        }
        if matches!(values[source.0 as usize].ty, ConcreteWireType::Trapdoor { .. }) &&
            output_root.availability.is_none()
        {
            let trapdoor_ty = values[source.0 as usize].ty.clone();
            let leaf_types = trapdoor_leaf_types(&trapdoor_ty)?;
            let mut ctx = root_context!();
            let mut leaves = Vec::with_capacity(6);
            for ty in &leaf_types {
                leaves.push(allocate_scratch_matrix(&mut ctx, ty, PhysicalEncoding::FullEval)?);
            }
            let leaves: [PhysicalValueId; 6] = leaves
                .try_into()
                .map_err(|_| "GPU trapdoor return has a wrong leaf count".to_owned())?;
            let returned =
                pack_trapdoor_leaves(&mut ctx, trapdoor_ty, leaves, PhysicalEncoding::FullEval)?;
            emit_trapdoor_leaf_copies(&mut ctx, source, returned)?;
            if let Some(&public) = ctx.trapdoor_public_ids.get(&source) {
                ctx.trapdoor_public_ids.insert(returned, public);
            }
            output_ids.insert(name.clone(), returned);
            continue;
        }
        if let Some(availability) = output_root.availability {
            // Canonical artifact encoding reads coefficients. Keep the Eval
            // producer alive for other readers and share one iNTT among
            // multiple artifact names for the same physical output.
            let export_source = if let Some(&coefficient) = coefficient_exports.get(&source) {
                coefficient
            } else if matches!(values[source.0 as usize].ty, ConcreteWireType::Trapdoor { .. }) {
                let mut ctx = root_context!();
                let coefficient = inverse_trapdoor_for_export(&mut ctx, source)?;
                coefficient_exports.insert(source, coefficient);
                coefficient
            } else {
                let matrix = values[source.0 as usize]
                    .ty
                    .matrix_type()
                    .ok_or_else(|| "GPU artifact output is not a matrix".to_owned())?
                    .clone();
                if values[source.0 as usize].encodings.as_ref() != [PhysicalEncoding::FullEval] {
                    return Err("GPU artifact export requires a full-Eval matrix source".into());
                }
                let storage = StorageRef::Scratch(
                    u32::try_from(values.len())
                        .map_err(|_| "too many GPU artifact stagings".to_owned())?,
                );
                let native = backend.allocate_physical_matrix(
                    &matrix,
                    device,
                    PhysicalEncoding::FullCoeff,
                )?;
                let (physical, resident) =
                    physical_matrix(&matrix, PhysicalEncoding::FullCoeff, storage, native)?;
                let coefficient = value_id(values.len())?;
                values.push(physical);
                owners.insert(coefficient, resident);
                let source_binding = register_bindings(&mut bindings, &values, source)?;
                let coefficient_binding = register_bindings(&mut bindings, &values, coefficient)?;
                let implementation = implementations
                    .register(GpuImplementation::ntt(true))
                    .map_err(str::to_owned)?;
                let index = u32::try_from(operations.len())
                    .map_err(|_| "too many GPU operations".to_owned())?;
                operations.push(CompiledGpuOp {
                    implementation,
                    arguments: Box::new([
                        KernelArg::Value(source),
                        KernelArg::U32(0),
                        KernelArg::Value(coefficient),
                        KernelArg::U32(0),
                        KernelArg::U32(source_binding),
                        KernelArg::U32(coefficient_binding),
                    ]),
                    outputs: Box::new([coefficient]),
                    device,
                    grid: [1; 3],
                    block: [1; 3],
                    shared_bytes: 0,
                    predecessors: all_predecessors(&producer, source),
                    body: None,
                });
                producer.insert(
                    coefficient,
                    vec![(ColumnRange { start: 0, end: matrix.columns }, index)],
                );
                coefficient_exports.insert(source, coefficient);
                coefficient
            };
            let public_export_source =
                if matches!(values[source.0 as usize].ty, ConcreteWireType::Trapdoor { .. }) {
                    let public = *trapdoor_public_ids.get(&source).ok_or_else(|| {
                        "GPU trapdoor artifact lacks paired public output".to_owned()
                    })?;
                    if let Some(&coefficient) = coefficient_exports.get(&public) {
                        Some(coefficient)
                    } else {
                        let mut ctx = root_context!();
                        let coefficient = inverse_matrix_for_export(&mut ctx, public)?;
                        coefficient_exports.insert(public, coefficient);
                        Some(coefficient)
                    }
                } else {
                    None
                };
            let artifact_type = ArtifactType::from_wire_type(&values[source.0 as usize].ty)
                .ok_or_else(|| "GPU artifact output has no artifact type".to_owned())?;
            let mut ctx = root_context!();
            emit_artifact_export(
                &mut ctx,
                &mut slots,
                &mut export_templates,
                name,
                &[(None, export_source)],
                artifact_type,
                availability,
                public_export_source,
            )?;
        }
        if matches!(values[source.0 as usize].ty, ConcreteWireType::Trapdoor { .. }) {
            let trapdoor_ty = values[source.0 as usize].ty.clone();
            let leaf_types = trapdoor_leaf_types(&trapdoor_ty)?;
            let mut ctx = root_context!();
            let mut leaves = Vec::with_capacity(6);
            for ty in &leaf_types {
                leaves.push(allocate_scratch_matrix(&mut ctx, ty, PhysicalEncoding::FullEval)?);
            }
            let leaves: [PhysicalValueId; 6] = leaves
                .try_into()
                .map_err(|_| "GPU trapdoor return has wrong leaf count".to_owned())?;
            let returned =
                pack_trapdoor_leaves(&mut ctx, trapdoor_ty, leaves, PhysicalEncoding::FullEval)?;
            emit_trapdoor_leaf_copies(&mut ctx, source, returned)?;
            if let Some(&public) = ctx.trapdoor_public_ids.get(&source) {
                ctx.trapdoor_public_ids.insert(returned, public);
            }
            output_ids.insert(name.clone(), returned);
            continue;
        }
        let source_value = &values[source.0 as usize];
        let matrix = source_value
            .ty
            .matrix_type()
            .ok_or_else(|| "GPU return output is not a matrix".to_owned())?
            .clone();
        if source_value.parts.len() != matrix.ring.crt_depth() ||
            source_value.encodings.as_ref() != [PhysicalEncoding::FullEval]
        {
            return Err("GPU return output needs all ordered full-Eval CRT parts".into());
        }
        // A whole plan-owned matrix is returned in place: its scratch storage
        // becomes return storage (which scratch sharing never reuses), so the
        // producer writes the result where the caller reads it.
        if source_value.parts.iter().all(|part| matches!(part.storage, StorageRef::Output(_))) {
            output_ids.insert(name.clone(), source);
            continue;
        }
        let owner = owners.get(&source).ok_or("GPU return source has no owner")?;
        if !input_ids.values().any(|id| *id == source) &&
            source_value.parts.iter().all(|part| {
                part.view.byte_offset == 0 && matches!(part.storage, StorageRef::Scratch(_))
            }) &&
            owner.storages().count() == source_value.parts.len()
        {
            let mut physical = source_value.clone();
            let relabel = |storage: StorageRef| match storage {
                StorageRef::Scratch(number) => StorageRef::Output(number),
                other => other,
            };
            for part in physical.parts.iter_mut() {
                part.storage = relabel(part.storage);
            }
            let storage = owner
                .storages()
                .map(|(storage, bound)| (relabel(*storage), bound.clone()))
                .collect::<BTreeMap<_, _>>();
            let ready = owner.ready_events().iter().cloned().collect::<Vec<_>>();
            let returned = GpuResidentValue::new(Arc::new(physical.clone()), storage, ready.into())
                .map_err(str::to_owned)?;
            values[source.0 as usize] = physical;
            owners.insert(source, Arc::new(returned));
            output_ids.insert(name.clone(), source);
            continue;
        }
        let storage = StorageRef::Output(
            u32::try_from(values.len()).map_err(|_| "too many GPU return storages".to_owned())?,
        );
        let native =
            backend.allocate_physical_matrix(&matrix, device, PhysicalEncoding::FullEval)?;
        let (physical, owner) =
            physical_matrix(&matrix, PhysicalEncoding::FullEval, storage, native)?;
        let returned = value_id(values.len())?;
        values.push(physical);
        owners.insert(returned, owner);
        let source_binding = register_bindings(&mut bindings, &values, source)?;
        let return_binding = register_bindings(&mut bindings, &values, returned)?;
        let implementation = implementations
            .register(GpuImplementation::matrix_copy_view())
            .map_err(str::to_owned)?;
        let predecessors = all_predecessors(&producer, source);
        let op_index =
            u32::try_from(operations.len()).map_err(|_| "too many GPU operations".to_owned())?;
        operations.push(CompiledGpuOp {
            implementation,
            arguments: Box::new([
                KernelArg::Value(source),
                KernelArg::U32(0),
                KernelArg::Value(returned),
                KernelArg::U32(0),
                KernelArg::U32(source_binding),
                KernelArg::U32(return_binding),
            ]),
            outputs: Box::new([returned]),
            device,
            grid: [1; 3],
            block: [1; 3],
            shared_bytes: 0,
            predecessors,
            body: None,
        });
        producer.insert(returned, vec![(ColumnRange { start: 0, end: matrix.columns }, op_index)]);
        output_ids.insert(name.clone(), returned);
    }
    let mut site_starts = BTreeMap::<usize, u32>::new();
    for template in &export_templates {
        if template.occurrence == 0 && site_starts.insert(template.slot, template.site).is_some() {
            return Err("GPU export sites start at the same physical slot".into());
        }
    }
    let starts = site_starts.into_iter().collect::<Vec<_>>();
    if !starts.is_empty() && starts[0].0 != 0 {
        return Err("GPU export slots have an unreserved prefix".into());
    }
    let mut slot_ranges = Vec::with_capacity(starts.len());
    for (index, &(start, site)) in starts.iter().enumerate() {
        let end = starts.get(index + 1).map_or(slots.len(), |next| next.0);
        let count = end
            .checked_sub(start)
            .filter(|count| *count > 0)
            .ok_or_else(|| "GPU export site has an empty or overlapping slot range".to_owned())?;
        let count = u64::try_from(count)
            .map_err(|_| "GPU export site occurrence count exceeds u64".to_owned())?;
        slot_ranges.push((site, count, 1));
    }
    for template in &export_templates {
        let &(start, _) = starts
            .iter()
            .find(|(_, site)| *site == template.site)
            .ok_or_else(|| "GPU export template has no site range".to_owned())?;
        let expected = start
            .checked_add(
                usize::try_from(template.occurrence)
                    .map_err(|_| "GPU export occurrence exceeds host address space".to_owned())?,
            )
            .ok_or_else(|| "GPU export occurrence slot overflows".to_owned())?;
        if template.slot != expected {
            return Err("GPU export occurrence is not at its reserved physical slot".into());
        }
    }
    if starts.is_empty() && !slots.is_empty() {
        return Err("GPU export slots have no planned site".into());
    }
    for import in &external_io_imports {
        if import.before_operation as usize >= operations.len() {
            return Err("GPU selected artifact import has no consuming Graph operation".into());
        }
        let selector = values
            .get(import.selector.0 as usize)
            .ok_or("GPU selected artifact import has no physical selector")?;
        if !matches!(selector.ty, ConcreteWireType::Int | ConcreteWireType::ConstantInt) ||
            !matches!(selector.encodings.as_ref(), [PhysicalEncoding::Signed(_)]) ||
            selector.parts.len() != 1
        {
            return Err("GPU selected artifact import selector is not a signed scalar".into());
        }
    }
    let protected = input_ids
        .values()
        .chain(output_ids.values())
        .chain(waves.iter().flat_map(|wave| wave.owner_bindings.keys()))
        .chain(import_templates.iter().map(|import| &import.destination))
        .chain(external_io_imports.iter().map(|import| &import.destination))
        .chain(external_io_loops.iter().flat_map(|body| {
            body.carried_ids.iter().chain(body.imports.iter().map(|import| &import.destination))
        }))
        .chain(indexed_tables.iter().flat_map(|table| table.candidates.iter().map(|(id, _)| id)))
        .chain(trapdoor_public_ids.iter().flat_map(|(secret, public)| [secret, public]))
        .copied()
        .collect::<BTreeSet<_>>();
    let replayed = waves
        .iter()
        .map(|wave| (wave.body_start, wave.body_end))
        .chain(external_io_loops.iter().map(|body| (body.body_start, body.body_end)))
        .collect::<Vec<_>>();
    share_scratch_allocations(&values, &mut owners, &operations, &replayed, &protected)?;
    let program = CompiledGpuProgram {
        values: values.into_boxed_slice(),
        implementations,
        operations: operations.into_boxed_slice(),
        bindings: bindings.into_boxed_slice(),
        export_slots: reserve_gpu_export_slots(slot_ranges).map_err(str::to_owned)?,
    };
    program.validate().map_err(str::to_owned)?;
    Ok(PhysicalFrame {
        program,
        owners,
        slots,
        device,
        input_ids,
        integer_input_owners,
        real_input_owners,
        bytes_input_owners,
        sample_seeds,
        hash_resources,
        real_owners,
        real_output_owners,
        indexed_tables,
        dynamic_export_resources,
        preimage_replays,
        trapdoor_public_ids,
        output_ids,
        export_templates,
        import_templates,
        control_resets,
        waves,
        external_io_loops,
        external_io_imports,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::MemoryArtifactStore,
        backend::{poly::cpu_backend, poly_gpu::gpu_backend_on},
        executor::{ExecutionConfig, execute_in_session},
        gpu_runtime::GpuRuntime,
        matrix::{CpuSmallMatrix, PolyMatrix, SmallPolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
                params::DCRTPolyParams,
                poly::DCRTPoly,
            },
        },
        session::SessionStore,
    };
    use mxx_dsl::{DslContext, HashTag, Int, Ring};
    use mxx_ir_core::IntExpr;

    fn download_trapdoor_leaf_for_oracle(
        runtime: &GpuRuntime,
        secret: &Arc<GpuResidentValue>,
        leaf: usize,
    ) -> DCRTPolyMatrix {
        let source = secret.physical();
        let ty = trapdoor_leaf_types(&source.ty).unwrap()[leaf].clone();
        let parts = source
            .parts
            .iter()
            .filter(|part| part.leaf as usize == leaf)
            .map(|part| {
                let mut part = part.clone();
                part.leaf = 0;
                part
            })
            .collect::<Vec<_>>();
        let view = Arc::new(PhysicalValue {
            ty: ConcreteWireType::Matrix(ty),
            encodings: Box::new([source.encodings[leaf].clone()]),
            parts: parts.into_boxed_slice(),
            integer_ranges: BTreeMap::new(),
        });
        let resident = Arc::new(secret.with_physical_view(view).unwrap());
        runtime.download_matrix(&RuntimeValue::Resident(resident)).unwrap()
    }

    fn download_compact_preimage_for_oracle(
        parameters: &GpuDCRTPolyParams,
        device: i32,
        cpu: &DCRTPolyParams,
        owner: &Arc<GpuResidentValue>,
        maximum_magnitude: u32,
    ) -> DCRTPolyMatrix {
        let matrix = owner.physical().ty.matrix_type().expect("preimage matrix type");
        let [part] = owner.physical().parts.as_ref() else {
            panic!("compact preimage has more than one physical part");
        };
        let [PhysicalEncoding::CompactCoeff { magnitude_bytes }] =
            owner.physical().encodings.as_ref()
        else {
            panic!("preimage output is not compact coefficient data");
        };
        let view = &part.view;
        let storage = owner.storage(part.storage).expect("compact owner storage");
        assert_eq!(view.origin.as_ref(), &[0, 0, 0, 0]);
        assert_eq!(
            view.extent.as_ref(),
            &[
                matrix.rows as u64,
                matrix.columns as u64,
                matrix.ring.ring_dimension() as u64,
                (*magnitude_bytes + 1) as u64,
            ]
        );
        let mut bytes = vec![0u8; storage.bytes as usize];
        parameters.download_device_bytes(device, storage.address, &mut bytes).unwrap();
        assert_eq!(view.element_bytes, 1);
        let width = view.extent[3] as usize;
        let modulus = cpu.modulus();
        let maximum = num_bigint::BigUint::from(maximum_magnitude);
        let mut entries = Vec::with_capacity(matrix.rows);
        for row in 0..matrix.rows {
            let mut columns = Vec::with_capacity(matrix.columns);
            for column in 0..matrix.columns {
                let mut coefficients = Vec::with_capacity(matrix.ring.ring_dimension() as usize);
                for coefficient in 0..matrix.ring.ring_dimension() as usize {
                    let offset = view.byte_offset as usize +
                        row * view.byte_strides[0] as usize +
                        column * view.byte_strides[1] as usize +
                        coefficient * view.byte_strides[2] as usize;
                    let packed = &bytes[offset..offset + width];
                    let magnitude = num_bigint::BigUint::from_bytes_le(&packed[1..]);
                    assert!(magnitude <= maximum);
                    let residue = match packed[0] {
                        0 if magnitude == num_bigint::BigUint::ZERO => magnitude,
                        1 => magnitude % modulus.as_ref(),
                        2 if magnitude != num_bigint::BigUint::ZERO => {
                            (modulus.as_ref() - (&magnitude % modulus.as_ref())) % modulus.as_ref()
                        }
                        tag => panic!("invalid compact coefficient sign {tag}"),
                    };
                    coefficients.push(residue);
                }
                columns.push(DCRTPoly::from_biguints(cpu, &coefficients));
            }
            entries.push(columns);
        }
        DCRTPolyMatrix::from_poly_vec(cpu, entries)
    }

    #[test]
    fn direct_static_gadget_matches_cpu_canonical_matrix() {
        let parameters = DCRTPolyParams::new(8, 2, 17, 2, None, None);
        let moduli = parameters.to_crt().0;
        let ring = mxx_ir_core::RingRef::new(mxx_ir_core::RingExpr::Explicit {
            crt_moduli: moduli.into_iter().map(IntExpr::from).collect(),
            ring_dimension: 8,
        })
        .resolve(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
        .expect("explicit ring");
        let per_tower = parameters.crt_bits().div_ceil(parameters.base_bits() as usize);
        for (digits, small) in
            [(per_tower, true), (per_tower, false), (parameters.modulus_digits(), false)]
        {
            let ty = ConcreteMatrixType { ring: ring.clone(), rows: 2, columns: 2 * digits };
            let value = ConstantMatrix::Gadget {
                base: IntExpr::from(1u64 << parameters.base_bits()),
                small,
            };
            let actual = encode_static_matrix(
                &ty,
                &value,
                &ParamEnv::default(),
                (!small).then_some(per_tower),
            )
            .expect("direct gadget encoding");
            let expected = if small {
                DCRTPolyMatrix::small_gadget_matrix(&parameters, ty.rows)
            } else {
                DCRTPolyMatrix::gadget_matrix(&parameters, ty.rows, Some(digits))
            };
            assert_eq!(actual, expected.to_compact_bytes());
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_compact_centered_rebase_matches_cpu_and_replay() {
        let device = detected_gpu_device_ids()[0];
        let source_cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let destination_cpu = DCRTPolyParams::new(32, 1, 17, 8, None, None);
        let source_moduli = source_cpu.to_crt().0;
        let destination_moduli = destination_cpu.to_crt().0;
        let source_gpu = GpuDCRTPolyParams::new(32, source_moduli.clone(), 8, None);
        let destination_gpu = GpuDCRTPolyParams::new(32, destination_moduli.clone(), 8, None);
        let source_ring =
            Ring::from_crt_moduli(source_moduli.into_iter().map(IntExpr::from).collect(), 32);
        let destination_ring =
            Ring::from_crt_moduli(destination_moduli.into_iter().map(IntExpr::from).collect(), 32);
        let input = source_ring.small_matrix_input("small", (1, 2), 7);
        let validated = DslContext::new("direct-compact-centered-rebase")
            .output("rebased", input.centered_rebase(&destination_ring))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let source_modulus = source_cpu.modulus();
        let signed = [-7i64, 0, 7, -1, 1];
        let columns = (0..2)
            .map(|column| {
                let coefficients = (0..32)
                    .map(|coefficient| {
                        let value = signed[(column * 32 + coefficient) % signed.len()];
                        if value < 0 {
                            source_modulus.as_ref() -
                                num_bigint::BigUint::from(value.unsigned_abs())
                        } else {
                            num_bigint::BigUint::from(value as u64)
                        }
                    })
                    .collect::<Vec<_>>();
                DCRTPoly::from_biguints(&source_cpu, &coefficients)
            })
            .collect::<Vec<_>>();
        let cpu_matrix = DCRTPolyMatrix::from_poly_vec_row(&source_cpu, columns);
        let cpu_small = CpuSmallMatrix::new(cpu_matrix, num_bigint::BigUint::from(7u8)).unwrap();
        let expected = cpu_small
            .centered_rebase(&destination_cpu)
            .unwrap()
            .to_canonical_coefficients()
            .unwrap();
        let payload = cpu_small.to_canonical_coefficients().unwrap();
        let native = GpuSmallMatrixOutputDescriptor::for_shape_in_domain(
            &source_gpu,
            1,
            2,
            num_bigint::BigUint::from(7u8),
            CoefficientBoundDomain::Global,
        )
        .and_then(|descriptor| descriptor.allocate())
        .unwrap();
        native
            .upload_canonical_coefficients_in_place(CoefficientBoundDomain::Global, &payload)
            .unwrap();
        let native = Arc::new(native);
        let binding = native.binding_descriptor().unwrap();
        let width = binding.magnitude_bytes + 1;
        let column_stride = 32 * width;
        let storage = StorageRef::Input(0);
        let input_wire = validated
            .source
            .root_scope()
            .nodes()
            .iter()
            .enumerate()
            .find_map(|(index, node)| {
                matches!(node.kind(), NodeKind::Input { name, .. } if name == "small").then_some(
                    WireRef { node: mxx_ir_core::types::NodeId(index as u64), port: Port(0) },
                )
            })
            .unwrap();
        let physical = PhysicalValue {
            ty: validated.root_scope().wire_types[&input_wire].clone(),
            encodings: Box::new([PhysicalEncoding::CompactCoeff {
                magnitude_bytes: binding.magnitude_bytes,
            }]),
            parts: Box::new([PhysicalPart {
                leaf: 0,
                storage,
                device,
                view: PhysicalView {
                    byte_offset: 0,
                    origin: Box::new([0, 0, 0, 0]),
                    extent: Box::new([1, 2, 32, width as u64]),
                    byte_strides: Box::new([
                        (2 * column_stride) as u64,
                        column_stride as u64,
                        width as u64,
                        1,
                    ]),
                    element_bytes: 1,
                },
            }]),
            integer_ranges: BTreeMap::new(),
        };
        assert_eq!(binding.payload_bytes, 2 * column_stride);
        let resident = Arc::new(
            GpuResidentValue::new(
                Arc::new(physical),
                BTreeMap::from([(
                    storage,
                    BoundStorage::from_small_matrix_payload(native).unwrap(),
                )]),
                Box::new([]),
            )
            .unwrap(),
        );
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([source_gpu, destination_gpu.clone()], [device]))
                .unwrap();
        let input = BTreeMap::from([("small".to_owned(), RuntimeValue::Resident(resident))]);
        let mut plan = runtime.plan(validated, &input).unwrap();
        let mut store = MemoryArtifactStore::default();
        let mut owners = Vec::new();
        for inputs in [input.clone(), input] {
            let result = runtime.execute(&mut plan, inputs, &mut store, rand::random()).unwrap();
            let output = result.output_value_for_test("rebased").unwrap();
            let RuntimeValue::Resident(owner) = output else {
                panic!("compact output is not GPU resident")
            };
            let [part] = owner.physical().parts.as_ref() else {
                panic!("compact output has multiple parts")
            };
            let storage = owner.storage(part.storage).unwrap();
            let mut actual = vec![0u8; storage.bytes as usize];
            destination_gpu.download_device_bytes(device, storage.address, &mut actual).unwrap();
            assert_eq!(actual, expected);
            owners.push(Arc::clone(owner));
        }
        assert!(Arc::ptr_eq(&owners[0], &owners[1]));
    }

    /// Runs only when a CUDA device is deliberately selected for validation.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_add_reuses_plan_owned_output() {
        let device = detected_gpu_device_ids()[0];
        let narrow = DCRTPolyParams::new(32, 1, 28, 8, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(32, 1, 50, 8, None, None).to_crt().0[0];
        let parameters = GpuDCRTPolyParams::new(32, vec![narrow, wide], 8, None);
        let ring = Ring::from_crt_moduli(
            parameters.to_crt().0.into_iter().map(IntExpr::from).collect(),
            parameters.ring_dimension(),
        );
        let left = ring.input("left", (1, 2));
        let right = ring.input("right", (1, 2));
        let validated = DslContext::new("direct-root-add")
            .output("sum", left + right)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let backend = gpu_backend_on([parameters], [device]);
        let mut inputs = BTreeMap::new();
        for (index, name) in ["left", "right"].into_iter().enumerate() {
            let wire = validated
                .source
                .root_scope()
                .nodes()
                .iter()
                .enumerate()
                .find_map(|(node_index, node)| match node.kind() {
                    NodeKind::Input { name: input_name, .. } if input_name == name => {
                        Some(WireRef {
                            node: mxx_ir_core::types::NodeId(node_index as u64),
                            port: Port(0),
                        })
                    }
                    _ => None,
                })
                .expect("declared input");
            let ty = validated.root_scope().wire_types[&wire].matrix_type().unwrap();
            let native =
                backend.allocate_physical_matrix(ty, device, PhysicalEncoding::FullEval).unwrap();
            let storage = StorageRef::Input(index as u32);
            let (_, resident) =
                physical_matrix(ty, PhysicalEncoding::FullEval, storage, native).unwrap();
            inputs.insert(name.to_owned(), RuntimeValue::Resident(resident));
        }
        let mut runtime = GpuRuntime::new(backend).unwrap();
        let mut plan = runtime.plan(validated, &inputs).unwrap();
        let mut store = MemoryArtifactStore::default();
        let first = runtime.execute(&mut plan, inputs.clone(), &mut store, [1; 32]).unwrap();
        let output_owner = |result: &crate::gpu_runtime::GpuExecutionResult| {
            let RuntimeValue::Matrix(matrix) = &result.output_value_for_test("sum").unwrap() else {
                panic!("matrix output");
            };
            Arc::clone(matrix.as_gpu().expect("GPU resident result"))
        };
        let first_owner = output_owner(&first);
        drop(first);
        let second = runtime.execute(&mut plan, inputs.clone(), &mut store, [2; 32]).unwrap();
        let second_owner = output_owner(&second);
        assert!(Arc::ptr_eq(&first_owner, &second_owner));
        let RuntimeValue::Resident(left_owner) = &inputs["left"] else {
            panic!("resident input");
        };
        assert!(!Arc::ptr_eq(&first_owner, left_owner));
        assert_eq!(plan.compiled_launch_count(), 2);
    }

    /// A chain of dependent doublings keeps only a few live intermediates, so
    /// its scratch matrices share allocations and still produce 2^k·x.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_dependent_chain_shares_scratch_allocations() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let parameters = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let steps = 8;
        let mut value = ring.uniform_residue((2, 2));
        let source = value.clone();
        for _ in 0..steps {
            value = value.clone() + value;
        }
        let validated = DslContext::new("direct-shared-scratch-chain")
            .output("source", source)
            .unwrap()
            .output("doubled", value)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([parameters], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let frame = plan.physical_frame_for_test();
        let scratch = frame
            .owners
            .iter()
            .filter(|(id, _)| {
                let physical = &frame.program.values[id.0 as usize];
                physical.encodings.as_ref() == [PhysicalEncoding::FullEval] &&
                    physical
                        .parts
                        .iter()
                        .all(|part| matches!(part.storage, StorageRef::Scratch(_)))
            })
            .collect::<Vec<_>>();
        let allocations = scratch
            .iter()
            .flat_map(|(_, owner)| owner.storages().map(|(_, bound)| bound.address))
            .collect::<BTreeSet<_>>();
        assert!(scratch.len() >= steps, "every doubling has a scratch result");
        assert!(allocations.len() < scratch.len(), "dependent intermediates share allocations");
        let mut store = MemoryArtifactStore::default();
        let result =
            runtime.execute(&mut plan, BTreeMap::new(), &mut store, rand::random()).unwrap();
        let source = runtime.download_matrix_output(&result.output("source").unwrap()).unwrap();
        let doubled = runtime.download_matrix_output(&result.output("doubled").unwrap()).unwrap();
        let scale =
            DCRTPoly::from_biguint_to_constant(&cpu, num_bigint::BigUint::from(1u32 << steps));
        assert_eq!(doubled, source.multiply_poly_out_of_place(&scale));
    }

    /// Concat pieces with a single elementwise or product writer and no
    /// other reader are written straight into their windows, and a whole
    /// returned matrix is returned in place, so only the shared piece is
    /// copied.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_concat_pieces_and_returns_are_written_in_place() {
        use crate::gpu_execution_plan::GpuNativePrimitive;
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let parameters = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let x = ring.uniform_residue((1, 1));
        let y = ring.uniform_residue((1, 1));
        let joined = mxx_dsl::Mat::concat(
            mxx_ir_core::node::ConcatAxis::Rows,
            vec![x.clone() * y.clone(), x.clone() + y.clone(), x.clone()],
        );
        let validated = DslContext::new("direct-concat-in-place")
            .output("x", x)
            .unwrap()
            .output("y", y)
            .unwrap()
            .output("joined", joined)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([parameters], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let frame = plan.physical_frame_for_test();
        let copies = frame
            .program
            .operations
            .iter()
            .filter(|op| {
                frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                    GpuNativePrimitive::MatrixCopyView
            })
            .map(|op| op.outputs.len())
            .collect::<Vec<_>>();
        assert_eq!(copies, vec![1], "only the shared piece x is copied");
        let mut store = MemoryArtifactStore::default();
        let result =
            runtime.execute(&mut plan, BTreeMap::new(), &mut store, rand::random()).unwrap();
        let x = runtime.download_matrix_output(&result.output("x").unwrap()).unwrap();
        let y = runtime.download_matrix_output(&result.output("y").unwrap()).unwrap();
        let joined = runtime.download_matrix_output(&result.output("joined").unwrap()).unwrap();
        let expected = (x.clone() * y.clone()).concat_rows(&[&(x.clone() + y), &x]);
        assert_eq!(joined, expected);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_trapdoor_sample_assembles_public_and_six_secret_leaves() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let parameters = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let digits = parameters.modulus_digits();
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let trapdoor = ring.sample_trapdoor(1, 4, 1u64 << 8, digits, 1_000_000);
        let validated = DslContext::new("direct-trapdoor-public")
            .output("public", trapdoor.public_matrix())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let backend = gpu_backend_on([parameters], [device]);
        let mut runtime = GpuRuntime::new(backend).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let mut store = MemoryArtifactStore::default();
        let first = runtime.execute(&mut plan, BTreeMap::new(), &mut store, [11; 32]).unwrap();
        let RuntimeValue::Matrix(first_public) = &first.output_value_for_test("public").unwrap()
        else {
            panic!("public matrix output");
        };
        let first_owner = Arc::clone(first_public.as_gpu().expect("resident public"));
        drop(first);
        let second = runtime.execute(&mut plan, BTreeMap::new(), &mut store, [12; 32]).unwrap();
        let RuntimeValue::Matrix(second_public) = &second.output_value_for_test("public").unwrap()
        else {
            panic!("public matrix output");
        };
        let second_owner = second_public.as_gpu().expect("resident public");
        assert!(Arc::ptr_eq(&first_owner, second_owner));
        assert_eq!(first_owner.wire_type().matrix_type().unwrap().columns, digits + 2);
        assert_eq!(plan.compiled_launch_count(), 2);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_trapdoor_export_pairs_public_and_secret() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let parameters = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let digits = parameters.modulus_digits();
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let trapdoor = ring.sample_trapdoor(1, 4, 1u64 << 8, digits, 1_000_000);
        let validated = DslContext::new("direct-trapdoor-secret-artifact")
            .transferred_trapdoor_output("secret", trapdoor)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([parameters], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let mut store = MemoryArtifactStore::default();
        let result = runtime.execute(&mut plan, BTreeMap::new(), &mut store, [13; 32]).unwrap();
        assert!(result.production_id.is_some());
        assert_eq!(result.artifact_handles["secret"].len(), 1);
        let RuntimeValue::Resident(secret) = &result.output_value_for_test("secret").unwrap()
        else {
            panic!("GPU trapdoor must remain resident");
        };
        assert_eq!(secret.physical().encodings.len(), 6);
        assert_eq!(secret.physical().parts.len(), 6);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_hash_sample_matches_cpu_and_rebinds_host_key() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let gpu = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let mut tag = HashTag::from(b"direct-hash/v1:".as_slice());
        tag.push("ordered");
        tag.push(IntExpr::from(9u8));
        tag.push(Int::constant(7));
        let hash = ring.hash_matrix(ring.bytes_input("key", 32), tag, (1, 1));
        let validated = DslContext::new("direct-hash-cpu-oracle")
            .output("hash", hash)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let key_a = RuntimeValue::Bytes(Arc::from([0x31u8; 32]));
        let key_b = RuntimeValue::Bytes(Arc::from([0xa7u8; 32]));
        let cpu_hash = |key: RuntimeValue| {
            let result = execute_in_session(
                &validated,
                &mut cpu_backend([cpu.clone()]),
                BTreeMap::from([("key".to_owned(), key)]),
                &mut MemoryArtifactStore::default(),
                [0x25; 32],
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::Matrix(matrix) = &result.outputs["hash"] else {
                panic!("CPU hash is a matrix");
            };
            matrix.as_cpu_full().unwrap().clone()
        };
        let expected_a = cpu_hash(key_a.clone());
        let expected_b = cpu_hash(key_b.clone());
        assert_ne!(expected_a, expected_b);
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu], [device])).unwrap();
        let mut plan =
            runtime.plan(validated, &BTreeMap::from([("key".to_owned(), key_a.clone())])).unwrap();
        let mut store = MemoryArtifactStore::default();
        let first = runtime
            .execute(&mut plan, BTreeMap::from([("key".to_owned(), key_a)]), &mut store, [0x41; 32])
            .unwrap();
        let actual_a = runtime.download_matrix_output(&first.output("hash").unwrap()).unwrap();
        drop(first);
        let second = runtime
            .execute(&mut plan, BTreeMap::from([("key".to_owned(), key_b)]), &mut store, [0x42; 32])
            .unwrap();
        assert_eq!(actual_a, expected_a);
        assert_eq!(
            runtime.download_matrix_output(&second.output("hash").unwrap()).unwrap(),
            expected_b
        );
        assert_eq!(plan.compiled_launch_count(), 2);
    }

    /// Hash integer families match the CPU transcript for one- and two-word
    /// moduli, and follow a rebound key.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_hash_int_family_matches_cpu_and_rebinds_host_key() {
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let gpu = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let context = DslContext::new("direct-hash-int-family");
        let key = ring.bytes_input("key", 32);
        let mut tag = HashTag::from(b"direct-hash-int/v1:".as_slice());
        tag.push(Int::constant(7));
        let narrow = context.hash_int_family(key.clone(), tag, 5, BigInt::from(1u64 << 32));
        let wide = context.hash_int_family(
            key,
            HashTag::from(b"wide".as_slice()),
            3,
            BigInt::from(1u8) << 70usize,
        );
        let validated = context
            .output("narrow", narrow)
            .unwrap()
            .output("wide", wide)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let cpu_hash = |key: RuntimeValue| {
            let result = execute_in_session(
                &validated,
                &mut cpu_backend([cpu.clone()]),
                BTreeMap::from([("key".to_owned(), key)]),
                &mut MemoryArtifactStore::default(),
                [0x25; 32],
                ExecutionConfig::default(),
            )
            .unwrap();
            ["narrow", "wide"].map(|name| {
                let RuntimeValue::IndexedFamily { values, .. } = &result.outputs[name] else {
                    panic!("CPU hash family is an integer family");
                };
                values
                    .iter()
                    .map(|value| match value {
                        RuntimeValue::Int(value) => value.clone(),
                        _ => panic!("CPU hash family member is an integer"),
                    })
                    .collect::<Vec<_>>()
            })
        };
        let key_a = RuntimeValue::Bytes(Arc::from([0x31u8; 32]));
        let key_b = RuntimeValue::Bytes(Arc::from([0xa7u8; 32]));
        let expected_a = cpu_hash(key_a.clone());
        let expected_b = cpu_hash(key_b.clone());
        assert_ne!(expected_a, expected_b);
        assert!(expected_a[0].iter().all(|value| value.bits() <= 32));
        assert!(expected_a[1].iter().any(|value| value.bits() > 64));
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let mut plan =
            runtime.plan(validated, &BTreeMap::from([("key".to_owned(), key_a.clone())])).unwrap();
        let mut store = MemoryArtifactStore::default();
        for (key, expected) in [(key_a, expected_a), (key_b, expected_b)] {
            let result = runtime
                .execute(
                    &mut plan,
                    BTreeMap::from([("key".to_owned(), key)]),
                    &mut store,
                    [0x41; 32],
                )
                .unwrap();
            for (name, expected) in ["narrow", "wide"].into_iter().zip(expected) {
                let actual =
                    runtime.download_integer_family_output(&result.output(name).unwrap()).unwrap();
                assert_eq!(actual, expected, "{name}");
            }
        }
    }

    /// Both integer matrix-vector product orientations match the CPU for a
    /// canonical hash matrix and a signed vector, across several row chunks.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_int_matrix_vector_products_match_cpu() {
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let gpu = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let context = DslContext::new("direct-int-matrix-vector");
        let key = ring.bytes_input("key", 32);
        let (rows, columns) = (130usize, 37usize);
        let matrix = context.hash_int_family(
            key.clone(),
            HashTag::from(b"matrix".as_slice()),
            rows * columns,
            BigInt::from(1u64 << 32),
        );
        let raw = context.hash_int_family(
            key,
            HashTag::from(b"vector".as_slice()),
            rows,
            BigInt::from(1u64 << 8),
        );
        let vector =
            mxx_dsl::parallel(rows, |index| Ok(raw.at(index).sub(Int::constant(128)))).unwrap();
        let validated = context
            .output("transposed", matrix.vector_matrix_product(&vector))
            .unwrap()
            .output("direct", matrix.matrix_vector_product(&vector))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let inputs =
            BTreeMap::from([("key".to_owned(), RuntimeValue::Bytes(Arc::from([0x5cu8; 32])))]);
        let expected = execute_in_session(
            &validated,
            &mut cpu_backend([cpu]),
            inputs.clone(),
            &mut MemoryArtifactStore::default(),
            [0x25; 32],
            ExecutionConfig::default(),
        )
        .unwrap();
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let mut plan = runtime.plan(validated, &inputs).unwrap();
        let result = runtime
            .execute(&mut plan, inputs, &mut MemoryArtifactStore::default(), [0x41; 32])
            .unwrap();
        for name in ["transposed", "direct"] {
            let RuntimeValue::IndexedFamily { values, .. } = &expected.outputs[name] else {
                panic!("CPU product is an integer family");
            };
            let expected = values
                .iter()
                .map(|value| match value {
                    RuntimeValue::Int(value) => value.clone(),
                    _ => panic!("CPU product member is an integer"),
                })
                .collect::<Vec<_>>();
            assert!(expected.iter().any(|value| value.sign() == num_bigint::Sign::Minus), "{name}");
            let actual =
                runtime.download_integer_family_output(&result.output(name).unwrap()).unwrap();
            assert_eq!(actual, expected, "{name}");
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_hash_small_decomposed_matches_cpu_per_crt_limb() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 2, 28, 8, None, None);
        let moduli = cpu.to_crt().0;
        let gpu = GpuDCRTPolyParams::new(32, moduli.clone(), 8, None);
        let digits = gpu.crt_bits().div_ceil(gpu.base_bits() as usize);
        let ring = Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), 32);
        let hash = ring.hash_small_decomposed(
            ring.bytes_input("key", 32),
            HashTag::from(b"direct-small-hash/v1".as_slice()),
            (digits, 1),
            1u64 << 8,
            digits,
        );
        let validated = DslContext::new("direct-hash-small-per-crt-limb")
            .output("hash", hash)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let key_a = RuntimeValue::Bytes(Arc::from([0x31u8; 32]));
        let key_b = RuntimeValue::Bytes(Arc::from([0xa7u8; 32]));
        let cpu_payload = |key: RuntimeValue| {
            let result = execute_in_session(
                &validated,
                &mut cpu_backend([cpu.clone()]),
                BTreeMap::from([("key".to_owned(), key)]),
                &mut MemoryArtifactStore::default(),
                [0x25; 32],
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::Matrix(matrix) = &result.outputs["hash"] else {
                panic!("CPU bounded hash is a compact matrix");
            };
            let compact = matrix.as_cpu_compact().expect("CPU bounded hash owner");
            assert_eq!(compact.bound_domain(), CoefficientBoundDomain::PerCrtLimb);
            compact.to_canonical_coefficients().unwrap()
        };
        let expected_a = cpu_payload(key_a.clone());
        let expected_b = cpu_payload(key_b.clone());
        assert_ne!(expected_a, expected_b);
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu.clone()], [device])).unwrap();
        let mut plan =
            runtime.plan(validated, &BTreeMap::from([("key".to_owned(), key_a.clone())])).unwrap();
        let mut store = MemoryArtifactStore::default();
        let mut returned = Vec::new();
        let mut copied = None;
        let download = |owner: &Arc<GpuResidentValue>| {
            let [part] = owner.physical().parts.as_ref() else {
                panic!("GPU bounded hash has more than one compact part");
            };
            let storage = owner.storage(part.storage).expect("compact payload storage");
            let mut actual = vec![0u8; storage.bytes as usize];
            gpu.download_device_bytes(device, storage.address, &mut actual).unwrap();
            actual
        };
        for (key, expected) in [(key_a, expected_a.clone()), (key_b, expected_b)] {
            let result = runtime
                .execute(
                    &mut plan,
                    BTreeMap::from([("key".to_owned(), key)]),
                    &mut store,
                    rand::random(),
                )
                .unwrap();
            let RuntimeValue::Resident(owner) = &result.output_value_for_test("hash").unwrap()
            else {
                panic!("GPU bounded hash is not resident");
            };
            assert!(matches!(
                owner.physical().encodings.as_ref(),
                [PhysicalEncoding::CompactCoeffPerCrtLimb { .. }]
            ));
            assert_eq!(download(owner), expected);
            returned.push(Arc::clone(owner));
            if copied.is_none() {
                copied = Some(runtime.copy_output(&result.output("hash").unwrap()).unwrap());
            }
        }
        assert!(Arc::ptr_eq(&returned[0], &returned[1]));
        // The explicit copy survives the replay that overwrote plan storage.
        let Some(RuntimeValue::Resident(copied)) = copied else {
            panic!("copied GPU bounded hash is not resident");
        };
        assert!(!Arc::ptr_eq(&copied, &returned[0]));
        assert_eq!(download(&copied), expected_a);
        assert_eq!(plan.compiled_launch_count(), 2);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_hash_loads_bytes_artifact_at_first_consumer() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let gpu = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let key_bytes = RuntimeValue::Bytes(Arc::from([0x5au8; 32]));
        let producer = DslContext::new("direct-bytes-artifact-producer")
            .cached_output("key", ring.bytes_input("key", 32))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut store = MemoryArtifactStore::default();
        let produced = execute_in_session(
            &producer,
            &mut cpu_backend([cpu.clone()]),
            BTreeMap::from([("key".to_owned(), key_bytes.clone())]),
            &mut store,
            [0x61; 32],
            ExecutionConfig::default(),
        )
        .unwrap();
        let production = produced.production_id.expect("bytes producer identity");
        let manifest = store.load_finalized_manifest(&production).unwrap();
        let key = ArtifactKey { production: production.clone(), name: "key".into(), index: None };
        let hash = ring.hash_matrix(
            ring.bytes_artifact_input(production.clone(), "key", 32, ArtifactAvailability::Cached),
            HashTag::from(b"direct-imported-key/v1".as_slice()),
            (1, 1),
        );
        let validated = DslContext::new("direct-bytes-artifact-consumer")
            .output("hash", hash)
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        assert_eq!(store.load_count(&key), 0);
        let first = runtime.execute(&mut plan, BTreeMap::new(), &mut store, [0x62; 32]).unwrap();
        assert_eq!(store.load_count(&key), 1);
        let first_matrix = runtime.download_matrix_output(&first.output("hash").unwrap()).unwrap();
        drop(first);
        let second = runtime.execute(&mut plan, BTreeMap::new(), &mut store, [0x63; 32]).unwrap();
        assert_eq!(store.load_count(&key), 2);
        assert_eq!(
            first_matrix,
            runtime.download_matrix_output(&second.output("hash").unwrap()).unwrap(),
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_preimage_sample_uses_bounded_device_retry() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let parameters = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let download_parameters = parameters.clone();
        let digits = parameters.modulus_digits();
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let trapdoor = ring.sample_trapdoor(1, 4, 1u64 << 8, digits, 1_000_000);
        let preimage = trapdoor.sample_preimage(ring.zero((1, 1)), (digits + 2, 1));
        let validated = DslContext::new("direct-preimage-retry")
            .output("public", trapdoor.public_matrix())
            .unwrap()
            .output("preimage", preimage)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([parameters], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let mut store = MemoryArtifactStore::default();
        let first =
            runtime.execute(&mut plan, BTreeMap::new(), &mut store, rand::random()).unwrap();
        let first_public = first.output_value_for_test("public").unwrap().clone();
        let first_preimage = first.output_value_for_test("preimage").unwrap().clone();
        drop(first);
        let frame = plan.physical_frame_for_test();
        let retry = frame
            .program
            .operations
            .iter()
            .find(|op| {
                frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                    crate::gpu_execution_plan::GpuNativePrimitive::LoopWhile
            })
            .expect("preimage retry loop");
        let gq = retry
            .body
            .as_ref()
            .expect("preimage retry body")
            .iter()
            .find(|op| {
                frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                    crate::gpu_execution_plan::GpuNativePrimitive::GqSample
            })
            .expect("GQ sample");
        let (KernelArg::Value(residual_id), KernelArg::Value(z_id)) =
            (&gq.arguments[1], &gq.arguments[3])
        else {
            panic!("GQ source and destination arguments");
        };
        let residual = runtime
            .download_matrix(&RuntimeValue::Resident(Arc::clone(&frame.owners[residual_id])))
            .unwrap();
        let z = runtime
            .download_matrix(&RuntimeValue::Resident(Arc::clone(&frame.owners[z_id])))
            .unwrap();
        let gadget = DCRTPolyMatrix::gadget_matrix(&cpu, 1, None);
        assert_eq!(&gadget * &z, residual, "raw GQ must solve the gadget equation");
        let public = runtime.download_matrix(&first_public).unwrap();
        let secret_id = *frame
            .trapdoor_public_ids
            .keys()
            .find(|id| frame.program.values[id.0 as usize].encodings.len() == 6)
            .expect("sampled six-leaf trapdoor");
        let secret = &frame.owners[&secret_id];
        let r = download_trapdoor_leaf_for_oracle(&runtime, secret, 0);
        let e = download_trapdoor_leaf_for_oracle(&runtime, secret, 1);
        let identity = DCRTPolyMatrix::identity(&cpu, digits, None);
        let trapdoor_basis = r.concat_rows(&[&e, &identity]);
        let public_id = frame.trapdoor_public_ids[&secret_id];
        let internal_public = runtime
            .download_matrix(&RuntimeValue::Resident(Arc::clone(&frame.owners[&public_id])))
            .unwrap();
        assert_eq!(public, internal_public, "returned public A must match its frame owner");
        let a_bar = public.slice_columns(0, 1);
        let middle = public.slice_columns(1, 2);
        let a1 = public.slice_columns(2, 2 + digits);
        assert_eq!(middle, DCRTPolyMatrix::identity(&cpu, 1, None));
        assert_eq!(
            &a1 + &(&(&a_bar * &r) + &e),
            gadget,
            "public gadget block must cancel Abar·R+E"
        );
        assert_eq!(
            &public * &trapdoor_basis,
            gadget,
            "sampled public matrix must pair with its trapdoor"
        );
        let body = retry.body.as_ref().unwrap();
        let correction = body
            .iter()
            .find(|op| {
                frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                    crate::gpu_execution_plan::GpuNativePrimitive::PreimageCorrection
            })
            .expect("preimage correction");
        let KernelArg::Value(candidate_id) = correction.arguments[0] else {
            panic!("correction candidate argument");
        };
        let KernelArg::Value(z_eval_id) = correction.arguments[6] else {
            panic!("correction gadget sample argument");
        };
        let copy = body
            .iter()
            .find(|op| op.outputs.as_ref() == [candidate_id])
            .expect("perturbation copy into candidate");
        let KernelArg::Value(perturb_id) = copy.arguments[0] else {
            panic!("perturbation copy source");
        };
        let download_intermediate = |id: PhysicalValueId| {
            runtime
                .download_matrix(&RuntimeValue::Resident(Arc::clone(&frame.owners[&id])))
                .unwrap()
        };
        let perturb = download_intermediate(perturb_id);
        let z_eval = download_intermediate(z_eval_id);
        let candidate_eval = download_intermediate(candidate_id);
        assert_eq!(
            candidate_eval,
            &perturb + &(&trapdoor_basis * &z_eval),
            "native correction must add [R; E; I]z to the perturbation"
        );
        assert_eq!(
            &public * &candidate_eval,
            DCRTPolyMatrix::zero(&cpu, 1, 1),
            "corrected candidate must solve the public relation"
        );
        let RuntimeValue::Resident(first_owner) = &first_preimage else {
            panic!("preimage output is not resident");
        };
        assert!(matches!(
            first_owner.physical().encodings.as_ref(),
            [PhysicalEncoding::CompactCoeff { .. }]
        ));
        let decoded = download_compact_preimage_for_oracle(
            &download_parameters,
            device,
            &cpu,
            first_owner,
            1_000_000,
        );
        assert_eq!(&public * &decoded, DCRTPolyMatrix::zero(&cpu, 1, 1));
        let second =
            runtime.execute(&mut plan, BTreeMap::new(), &mut store, rand::random()).unwrap();
        let RuntimeValue::Resident(second_owner) =
            second.output_value_for_test("preimage").unwrap()
        else {
            panic!("preimage output is not resident");
        };
        assert!(Arc::ptr_eq(first_owner, second_owner), "replay must reuse plan-owned output");
        let second_decoded = download_compact_preimage_for_oracle(
            &download_parameters,
            device,
            &cpu,
            second_owner,
            1_000_000,
        );
        let second_public =
            runtime.download_matrix(second.output_value_for_test("public").unwrap()).unwrap();
        assert_eq!(&second_public * &second_decoded, DCRTPolyMatrix::zero(&cpu, 1, 1));
        drop(second);
        assert_eq!(plan.compiled_launch_count(), 2);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_preimage_sample_tiled_columns_match_public_relation() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = cpu.to_crt().0[0];
        let parameters = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let download_parameters = parameters.clone();
        let digits = parameters.modulus_digits();
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);
        let trapdoor = ring.sample_trapdoor(1, 4, 1u64 << 8, digits, 1_000_000);
        let preimage = trapdoor.sample_preimage(ring.zero((1, 2)), (digits + 2, 2));
        let validated = DslContext::new("direct-preimage-tiled")
            .output("public", trapdoor.public_matrix())
            .unwrap()
            .output("preimage", preimage)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        // Each one-column tile runs on its own logical device when several
        // exist (`MXX_GPU_LOGICAL_DEVICES=0,0`).
        let devices = detected_gpu_device_ids();
        let mut runtime = GpuRuntime::new(gpu_backend_on([parameters], devices.clone())).unwrap();
        let mut plan =
            runtime.plan_with_fixed_columns_for_test(validated, &BTreeMap::new(), 1).unwrap();
        assert_eq!(plan.physical_frame_for_test().preimage_replays.len(), 2);
        let mut store = MemoryArtifactStore::default();
        let first =
            runtime.execute(&mut plan, BTreeMap::new(), &mut store, rand::random()).unwrap();
        let public =
            runtime.download_matrix(first.output_value_for_test("public").unwrap()).unwrap();
        let RuntimeValue::Resident(first_owner) = first.output_value_for_test("preimage").unwrap()
        else {
            panic!("preimage output is not resident");
        };
        let first_owner = Arc::clone(first_owner);
        let decoded = download_compact_preimage_for_oracle(
            &download_parameters,
            device,
            &cpu,
            &first_owner,
            1_000_000,
        );
        drop(first);
        let frame = plan.physical_frame_for_test();
        let retries = frame
            .program
            .operations
            .iter()
            .filter(|op| {
                frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                    crate::gpu_execution_plan::GpuNativePrimitive::LoopWhile
            })
            .collect::<Vec<_>>();
        assert_eq!(retries.len(), 2);
        let tile_devices = retries.iter().map(|retry| retry.device).collect::<BTreeSet<_>>();
        assert_eq!(tile_devices.len(), devices.len().min(2));
        for (column, retry) in retries.iter().enumerate() {
            let body = retry.body.as_ref().expect("preimage tile retry body");
            let correction = body
                .iter()
                .find(|op| {
                    frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                        crate::gpu_execution_plan::GpuNativePrimitive::PreimageCorrection
                })
                .expect("preimage tile correction");
            let cutoff = body
                .iter()
                .find(|op| {
                    frame.program.implementations.resolve(op.implementation).unwrap().primitive ==
                        crate::gpu_execution_plan::GpuNativePrimitive::PreimageCutoff
                })
                .expect("preimage tile cutoff");
            let KernelArg::Value(candidate_eval_id) = correction.arguments[0] else {
                panic!("preimage tile correction candidate");
            };
            let KernelArg::Value(candidate_coeff_id) = cutoff.arguments[1] else {
                panic!("preimage tile cutoff candidate");
            };
            let candidate_eval = runtime
                .download_matrix(&RuntimeValue::Resident(Arc::clone(
                    &frame.owners[&candidate_eval_id],
                )))
                .unwrap();
            assert_eq!(
                &public * &candidate_eval,
                DCRTPolyMatrix::zero(&cpu, 1, 1),
                "tile {column} candidate must solve the public relation"
            );
            let candidate_coeff = runtime
                .download_matrix(&RuntimeValue::Resident(Arc::clone(
                    &frame.owners[&candidate_coeff_id],
                )))
                .unwrap();
            assert_eq!(
                decoded.slice_columns(column, column + 1),
                candidate_coeff,
                "tile {column} publication must preserve its accepted candidate"
            );
        }
        assert_eq!(&public * &decoded, DCRTPolyMatrix::zero(&cpu, 1, 2));
        let second =
            runtime.execute(&mut plan, BTreeMap::new(), &mut store, rand::random()).unwrap();
        let RuntimeValue::Resident(second_owner) =
            second.output_value_for_test("preimage").unwrap()
        else {
            panic!("replayed preimage output is not resident");
        };
        assert!(Arc::ptr_eq(&first_owner, second_owner), "replay must reuse plan-owned output");
        let second_decoded = download_compact_preimage_for_oracle(
            &download_parameters,
            device,
            &cpu,
            second_owner,
            1_000_000,
        );
        // The replay samples a new trapdoor from its new nonce.
        let second_public =
            runtime.download_matrix(second.output_value_for_test("public").unwrap()).unwrap();
        assert_eq!(&second_public * &second_decoded, DCRTPolyMatrix::zero(&cpu, 1, 2));
        assert_eq!(plan.compiled_launch_count(), 2);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_public_gadget_preimage_matches_relation_and_bound() {
        let device = detected_gpu_device_ids()[0];
        let cpu = DCRTPolyParams::new(32, 2, 28, 8, None, None);
        let moduli = cpu.to_crt().0;
        let parameters = GpuDCRTPolyParams::new(32, moduli.clone(), 8, None);
        let download_parameters = parameters.clone();
        let digits = parameters.modulus_digits();
        let ring = Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), 32);
        let trapdoor = ring.gadget_trapdoor(1, 1u64 << 8, digits);
        let target = ring.identity(1);
        let preimage = trapdoor.sample_preimage(target.clone(), (digits, 1));
        let validated = DslContext::new("direct-public-gadget-preimage")
            .output("public", trapdoor.public_matrix())
            .unwrap()
            .output("target", target)
            .unwrap()
            .output("preimage", preimage)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut runtime = GpuRuntime::new(gpu_backend_on([parameters], [device])).unwrap();
        let mut plan = runtime.plan(validated, &BTreeMap::new()).unwrap();
        let mut store = MemoryArtifactStore::default();
        let mut previous = None;
        for _ in 0..2 {
            let result =
                runtime.execute(&mut plan, BTreeMap::new(), &mut store, rand::random()).unwrap();
            let public =
                runtime.download_matrix(&result.output_value_for_test("public").unwrap()).unwrap();
            let target =
                runtime.download_matrix(&result.output_value_for_test("target").unwrap()).unwrap();
            let RuntimeValue::Resident(owner) = &result.output_value_for_test("preimage").unwrap()
            else {
                panic!("public gadget preimage is not GPU resident");
            };
            if let Some(previous) = previous.replace(Arc::clone(owner)) {
                assert!(Arc::ptr_eq(&previous, owner));
            }
            let decoded = download_compact_preimage_for_oracle(
                &download_parameters,
                device,
                &cpu,
                owner,
                128,
            );
            assert_eq!(&public * &decoded, target);
        }
    }

    fn run_direct_artifact_import_smoke(cached_output: bool) {
        let device = detected_gpu_device_ids()[0];
        let parameters = DCRTPolyParams::new(32, 1, 28, 8, None, None);
        let modulus = parameters.to_crt().0[0];
        let gpu_parameters = GpuDCRTPolyParams::new(32, vec![modulus], 8, None);
        let ring = Ring::from_crt_moduli(vec![IntExpr::from(modulus)], 32);

        let producer = DslContext::new("direct-artifact-producer")
            .cached_output("stored", ring.identity(1))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .unwrap();
        let mut store = MemoryArtifactStore::default();
        let produced = execute_in_session(
            &producer,
            &mut cpu_backend([parameters]),
            BTreeMap::new(),
            &mut store,
            [0x51; 32],
            ExecutionConfig::default(),
        )
        .unwrap();
        let production = produced.production_id.expect("producer identity");
        let manifest = store.load_finalized_manifest(&production).unwrap();
        let imported =
            ring.artifact_input(production.clone(), "stored", (1, 1), ArtifactAvailability::Cached);
        let consumer = DslContext::new("direct-artifact-consumer");
        let sum = imported + ring.zero((1, 1));
        let consumer = if cached_output {
            consumer.cached_output("sum", sum).unwrap()
        } else {
            consumer.output("sum", sum).unwrap()
        };
        let consumer = consumer
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production.clone(), manifest)]),
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .unwrap();
        let key = ArtifactKey { production, name: "stored".into(), index: None };
        let mut runtime = GpuRuntime::new(gpu_backend_on([gpu_parameters], [device])).unwrap();
        let mut plan = runtime.plan(consumer, &BTreeMap::new()).unwrap();
        assert_eq!(store.load_count(&key), 0);
        let result = runtime.execute(&mut plan, BTreeMap::new(), &mut store, [0x52; 32]).unwrap();
        assert_eq!(store.load_count(&key), 1);
        if cached_output {
            assert!(result.production_id.is_some());
            assert_eq!(result.artifact_handles["sum"].len(), 1);
        }
    }

    /// The plan is store-free; the selected matrix is read at its first
    /// consuming GPU region and uploaded to the plan-time owner.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_add_imports_one_artifact_at_first_consumer() {
        run_direct_artifact_import_smoke(false);
    }

    /// The import and cached export share one direct Graph with asynchronous
    /// raw fragments and canonical artifact commit.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn direct_add_imports_and_exports_cached_matrix() {
        run_direct_artifact_import_smoke(true);
    }
}
