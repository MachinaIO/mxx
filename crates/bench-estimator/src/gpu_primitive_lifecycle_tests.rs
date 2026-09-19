//! End-to-end coverage for the ordinary primitive warmup lifecycle.
//!
//! This module deliberately starts at the public DSL.  A list of enum values or
//! a hand-written provider table is not evidence that a primitive is connected
//! to production lowering.  The test below builds one validated graph, asks the
//! real GPU measurement provider to profile every reachable family, freezes the
//! resulting plan, and executes that same graph through the fixed-plan entry
//! point.

#![cfg(feature = "gpu")]

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    sync::atomic::Ordering,
    time::Duration,
};

use mxx_dsl::{DslContext, Family, Ring};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix, IndexRange},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::poly::{
    PolyParams,
    dcrt::{
        gpu::{GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync, gpu_memory_info},
        params::DCRTPolyParams,
    },
};
use mxx_runtime::{
    Backend, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::{GpuWarmupProvenance, poly::cpu_backend, poly_gpu::gpu_backend_on},
    executor::execute_with_gpu_plan,
    gpu_column_policy::{
        CanonicalWarmupProfileDomain, WarmupMeasurementKind, canonical_warmup_profile_domain,
    },
    gpu_execution_plan::{GpuDeviceBudget, GpuLayout, GpuPlanContract, LayoutId},
    gpu_warmup::{GpuProfileProvenance, GpuStageCostModel, GpuValidatedWarmupConfig},
    transcript::SamplingMode,
};
use num_bigint::{BigInt, Sign};

use crate::{gpu::GpuNodeMeasurementBackend, harness::MeasurementHarnessConfig};

fn layouts(graph: &mxx_ir_core::ValidatedGraph) -> Vec<GpuLayout> {
    let mut by_type = BTreeMap::<ConcreteMatrixType, LayoutId>::new();
    for scope in graph.scopes.values() {
        for wire_type in scope.wire_types.values() {
            if let Some(matrix) = wire_type.matrix_type() {
                let next = by_type.len() as LayoutId + 1;
                by_type.entry(matrix.clone()).or_insert(next);
            }
        }
    }
    by_type
        .into_iter()
        .map(|(matrix, id)| GpuLayout {
            id,
            columns: matrix.columns,
            rows: matrix.rows,
            ring_dimension: matrix.ring_dimension,
            representation: format!("{:?}", ConcreteWireType::Matrix(matrix)),
            instance_device_stride: 0,
            owner_intervals: Vec::new(),
        })
        .collect()
}

fn config(
    graph: &mxx_ir_core::ValidatedGraph,
    parameters: &GpuDCRTPolyParams,
    device: i32,
) -> GpuValidatedWarmupConfig {
    GpuValidatedWarmupConfig {
        contract: GpuPlanContract {
            graph_specification_hash: [0; 32],
            backend_identity: "dsl-primitive-lifecycle".into(),
            logical_to_physical_devices: vec![device as usize],
            device_budgets: vec![GpuDeviceBudget {
                device: 0,
                device_bytes: gpu_memory_info(device).expect("query GPU memory").total as u64,
                pinned_host_bytes: u64::MAX,
                host_bytes: u64::MAX,
            }],
            shape_contract_hash: [0; 32],
            backend_revision: "dsl-primitive-lifecycle".into(),
        },
        layouts: layouts(graph),
        default_tile_widths: vec![1, 2],
        default_cost: vec![GpuStageCostModel::default()],
        default_implementation_variant: "dsl-primitive-lifecycle".into(),
        profiles: BTreeMap::new(),
        effective_operation_identities: BTreeMap::new(),
        effective_operations: BTreeMap::new(),
        storage_descriptors: BTreeMap::new(),
        active_crt_towers: parameters.modulus_digits(),
        crt_limb_bytes: 8,
        max_parallel_instances: NonZeroUsize::new(1).expect("non-zero wave limit"),
    }
}

/// Construct the ordinary primitive inventory through the public DSL.
///
/// The returned graph intentionally keeps each operation as a named output so
/// liveness cannot remove a primitive before validation.  The test does not
/// assert against a supplemental list of expected enum entries: it derives the
/// set of dispatched domains from the provider's production records and checks
/// that every validated non-structural node has one measured record.
fn ordinary_graph(parameters: &GpuDCRTPolyParams) -> mxx_ir_core::ValidatedGraph {
    let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
    let ring = Ring::new(modulus.clone(), parameters.ring_dimension() as usize);
    let context = DslContext::new("dsl-ordinary-primitive-lifecycle");
    let key = ring.bytes_input("hash-key", 32);

    let zero = ring.zero((1, 1));
    let identity = ring.identity(1);
    let unit_row = ring.constant((1, 1), ConstantMatrix::UnitRow { index: 0.into() });
    let unit_column = ring.constant((1, 1), ConstantMatrix::UnitColumn { index: 0.into() });
    let gadget = ring.constant(
        (1, parameters.modulus_digits()),
        ConstantMatrix::Gadget { base: 256.into(), small: false },
    );
    let power =
        ring.constant((1, 1), ConstantMatrix::PowerOfBase { base: 256.into(), exponent: 1.into() });
    let rotation = ring.constant((1, 1), ConstantMatrix::Rotation { exponent: 1.into() });
    let polynomial = ring.polynomial([1.into(), 2.into()]);

    let residue = ring.uniform_residue((1, 2));
    let interval = ring.uniform_interval((1, 2), -1, 1);
    let gaussian = ring.gaussian((1, 2), 1, 4);
    let hash = ring.hash_matrix(key.clone(), b"ordinary-hash".as_slice(), (1, 2));
    let hash_decomposed = ring.hash_decomposed(
        key.clone(),
        b"ordinary-hash-decomposed".as_slice(),
        (parameters.modulus_digits(), 1),
        256,
        parameters.modulus_digits(),
    );
    let hash_small = ring.hash_small_decomposed(
        key,
        b"ordinary-hash-small".as_slice(),
        (parameters.crt_bits().div_ceil(parameters.base_bits() as usize), 1),
        256,
        parameters.crt_bits().div_ceil(parameters.base_bits() as usize),
    );

    let add = interval.clone() + gaussian.clone();
    let sub = interval.clone() - gaussian.clone();
    let gaussian_rhs = gaussian.clone().transpose();
    let multiply = interval.clone() * gaussian_rhs.clone();
    let scale = interval.clone() * 2;
    let negate = -interval.clone();
    let accumulate = mxx_dsl::Mat::multi_row_gemm_accumulate(
        vec![(1, interval.clone(), gaussian_rhs)],
        Some(zero.clone()),
    );
    let small_rhs = ring
        .uniform_interval(
            (1, parameters.crt_bits().div_ceil(parameters.base_bits() as usize)),
            -1,
            1,
        )
        .mul_small_rhs(hash_small);

    let automorphism = interval.clone().ring_automorphism(1);
    let modulus_switch = interval.clone().modulus_switch(modulus.clone());
    let modulus_reduce = interval.clone().reduce_modulus(modulus.clone());
    let rns_source = Ring::new(BigInt::from(131_009u64), parameters.ring_dimension() as usize)
        .uniform_interval((1, 2), -1, 1);
    let centered_rebase = rns_source.clone().centered_rebase(modulus.clone());
    let block_mod_switch =
        interval.clone().block_mod_switch(130_817u64, vec![131_009, 130_817], 3u64);
    let rns_up = rns_source.rns_mod_up(modulus.clone(), vec![131_009], 1, true);
    let rns_down = rns_up.clone().rns_mod_down(131_009u64, vec![131_009, 130_817], 2);
    let crt = mxx_dsl::Mat::crt_recompose(
        vec![interval.clone()],
        vec![2.into()],
        vec![1.into()],
        modulus.into(),
    );

    let transposed = interval.clone().transpose();
    let sliced = interval.clone().slice(Some(IndexRange { start: 0.into(), end: 1.into() }), None);
    let tensor = interval.clone().tensor(gaussian.clone());
    let concat_rows =
        mxx_dsl::Mat::concat(ConcatAxis::Rows, vec![interval.clone(), gaussian.clone()]);
    let concat_columns =
        mxx_dsl::Mat::concat(ConcatAxis::Columns, vec![interval.clone(), gaussian.clone()]);
    let concat_diagonal =
        mxx_dsl::Mat::concat(ConcatAxis::Diagonal, vec![interval.clone(), gaussian.clone()]);

    let lifted = mxx_dsl::Int::constant(3).lift_to_constant_polynomial(ring.matrix_type((1, 1)));
    let coefficients = Family::pack(
        (0..parameters.ring_dimension() as usize)
            .map(|index| mxx_dsl::Int::constant(index as u64 + 1))
            .collect(),
    )
    .expect("coefficient family");
    let imported = ring.from_coefficients(&coefficients);
    let values = imported.coefficients();
    let coefficient_bits = parameters.modulus().bits() as usize;
    let bits = Family::pack(
        (0..parameters.ring_dimension() as usize * coefficient_bits)
            .map(|index| mxx_dsl::Bool::constant(index % coefficient_bits == 0))
            .collect(),
    )
    .expect("coefficient bits");
    let packed = ring.pack_polynomial_coefficients(bits, coefficient_bits);
    let extracted = imported.clone().extract_coefficient(0);
    let decoded = imported.clone().threshold_decode_ints(2, 1);

    let trapdoor = ring.sample_trapdoor(1, 1, 256, parameters.modulus_digits(), 10_000_000);
    let trapdoor_public = trapdoor.public_matrix();
    let preimage = trapdoor.sample_preimage(zero.clone(), (parameters.modulus_digits() + 2, 1));
    let decomposed = interval.clone().decompose(256, parameters.modulus_digits());
    let decomposed_small = interval
        .clone()
        .small_decompose(256, parameters.crt_bits().div_ceil(parameters.base_bits() as usize));

    let context = context
        .output("constant-zero", zero)
        .expect("zero")
        .output("constant-identity", identity)
        .expect("identity")
        .output("constant-unit-row", unit_row)
        .expect("unit row")
        .output("constant-unit-column", unit_column)
        .expect("unit column")
        .output("constant-gadget", gadget)
        .expect("gadget")
        .output("constant-power", power)
        .expect("power")
        .output("constant-rotation", rotation)
        .expect("rotation")
        .output("constant-polynomial", polynomial)
        .expect("polynomial")
        .output("uniform-residue", residue)
        .expect("residue")
        .output("uniform-interval", interval)
        .expect("interval")
        .output("gaussian", gaussian)
        .expect("gaussian")
        .output("hash", hash)
        .expect("hash")
        .output("hash-decomposed", hash_decomposed)
        .expect("decomposed hash")
        .output("add", add)
        .expect("add")
        .output("sub", sub)
        .expect("sub")
        .output("multiply", multiply)
        .expect("multiply")
        .output("scale", scale)
        .expect("scale")
        .output("negate", negate)
        .expect("negate")
        .output("accumulate", accumulate)
        .expect("accumulate")
        .output("small-rhs", small_rhs)
        .expect("small rhs")
        .output("automorphism", automorphism)
        .expect("automorphism")
        .output("modulus-switch", modulus_switch)
        .expect("modulus switch")
        .output("modulus-reduce", modulus_reduce)
        .expect("modulus reduce")
        .output("centered-rebase", centered_rebase)
        .expect("centered rebase")
        .output("block-mod-switch", block_mod_switch)
        .expect("block mod switch")
        .output("rns-up", rns_up)
        .expect("rns up")
        .output("rns-down", rns_down)
        .expect("rns down")
        .output("crt", crt)
        .expect("crt")
        .output("transpose", transposed)
        .expect("transpose")
        .output("slice", sliced)
        .expect("slice")
        .output("tensor", tensor)
        .expect("tensor")
        .output("concat-rows", concat_rows)
        .expect("rows")
        .output("concat-columns", concat_columns)
        .expect("columns")
        .output("concat-diagonal", concat_diagonal)
        .expect("diagonal")
        .output("lift", lifted)
        .expect("lift")
        .output("imported", imported)
        .expect("imported")
        .output("values", values)
        .expect("values")
        .output("packed", packed)
        .expect("packed")
        .output("extracted", extracted)
        .expect("extracted")
        .output("decoded", decoded[0].clone())
        .expect("decoded")
        .output("trapdoor-public", trapdoor_public)
        .expect("trapdoor public")
        .output("preimage", preimage)
        .expect("preimage")
        .output("gadget-decompose", decomposed)
        .expect("gadget decompose")
        .output("gadget-decompose-small", decomposed_small)
        .expect("small gadget decompose")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("validated DSL graph");
    context
}

#[test]
#[serial_test::serial(gpu_context)]
fn dsl_ordinary_primitives_complete_measured_fixed_lifecycle() {
    let device = detected_gpu_device_ids()
        .into_iter()
        .next()
        .expect("ordinary primitive lifecycle requires a detected GPU");
    let parameters = GpuDCRTPolyParams::new(8, vec![131_009, 130_817], 8, None);
    let graph = ordinary_graph(&parameters);
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let source_parameters = GpuDCRTPolyParams::new_with_gpu(
        8,
        vec![131_009],
        8,
        vec![device],
        None,
        Some(&parameters),
        None,
    );
    let block_parameters = GpuDCRTPolyParams::new_with_gpu(
        8,
        vec![130_817],
        8,
        vec![device],
        None,
        Some(&parameters),
        None,
    );
    let mut provider = GpuNodeMeasurementBackend::new(
        vec![(
            gpu_backend_on(
                [source_parameters, block_parameters.clone(), parameters.clone()],
                [device],
            ),
            device,
        )],
        harness,
    );
    let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_from_validated_with_provider(
        &graph,
        &config(&graph, &parameters, device),
        &mut provider,
    )
    .expect("every DSL primitive must be measured by the production provider");

    assert!(provider.warmup_measurement_call_count() > 0);
    assert!(
        provider
            .warmup_measurement_provenances()
            .iter()
            .all(|provenance| *provenance == GpuWarmupProvenance::ProductionEquivalent)
    );
    assert!(provider.warmup_dispatch_records().iter().all(|record| {
        record.range.start < record.range.end &&
            matches!(
                record.measurement,
                WarmupMeasurementKind::GpuMeasured | WarmupMeasurementKind::HostMeasured
            )
    }));
    let measured_domains = provider
        .warmup_dispatch_records()
        .iter()
        .map(|record| record.profile_domain)
        .collect::<BTreeSet<_>>();
    let validated_domains = graph
        .scopes
        .values()
        .flat_map(|scope| scope.execution_order.iter())
        .map(|node| match canonical_warmup_profile_domain(node.kind()) {
            // Fixed execution lowers every sampler to the metadata-bearing
            // batched preimage production path, including a single instance.
            CanonicalWarmupProfileDomain::PreimageSample => {
                CanonicalWarmupProfileDomain::FusedPreimageBatch
            }
            domain => domain,
        })
        .collect::<BTreeSet<CanonicalWarmupProfileDomain>>();
    assert!(
        validated_domains.is_subset(&measured_domains),
        "validated DSL domains missing production measurements: {:?}",
        validated_domains.difference(&measured_domains).collect::<Vec<_>>()
    );
    assert!(
        warmup
            .report
            .stages
            .iter()
            .all(|stage| stage.provenance != GpuProfileProvenance::ConservativeEstimate)
    );
    assert!(warmup.report.predicted_seconds.is_finite());
    assert!(warmup.report.predicted_seconds > 0.0);

    let measured_calls = provider.warmup_measurement_call_count();
    let call_counter = provider.warmup_measurement_counter();
    drop(provider);
    gpu_device_sync();

    let source_parameters = GpuDCRTPolyParams::new_with_gpu(
        8,
        vec![131_009],
        8,
        vec![device],
        None,
        Some(&parameters),
        None,
    );
    let mut backend =
        gpu_backend_on([source_parameters, block_parameters, parameters.clone()], [device]);
    backend.select_operation([0xA7; 32]).expect("select operation");
    warmup.plan.contract = backend
        .gpu_runtime_contract(
            &graph,
            &BTreeMap::from([("hash-key".to_owned(), RuntimeValue::Bytes(vec![0x57; 32]))]),
        )
        .expect("runtime contract")
        .expect("GPU contract");
    warmup.plan.validate().expect("frozen plan validation");
    let output = execute_with_gpu_plan(
        &graph,
        &warmup.plan,
        &mut backend,
        BTreeMap::from([("hash-key".to_owned(), RuntimeValue::Bytes(vec![0x57; 32]))]),
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("fixed execution must use the frozen measured plan");

    assert_eq!(output.outputs.len(), graph.source.outputs().len());
    assert!(
        output
            .outputs
            .values()
            .all(|value| { !matches!(value, RuntimeValue::Bytes(bytes) if bytes.is_empty()) })
    );
    let single_device_constant_count = warmup
        .plan
        .nodes
        .iter()
        .filter(|node| {
            node.effective_operation ==
                mxx_runtime::gpu_column_policy::EffectiveGpuOperation::SingleDeviceConstant
        })
        .count();
    assert!(
        single_device_constant_count >= 3,
        "all single-device constant variants must be frozen"
    );
    assert_eq!(
        warmup
            .plan
            .nodes
            .iter()
            .filter(|node| {
                node.effective_operation ==
                    mxx_runtime::gpu_column_policy::EffectiveGpuOperation::TrapdoorSample
            })
            .count(),
        1,
        "trapdoor sampling must be frozen"
    );
    assert_eq!(
        backend.fixed_single_device_constant_call_count(),
        single_device_constant_count,
        "single-device constants must use the metadata-bearing fixed path"
    );
    assert_eq!(
        backend.fixed_trapdoor_call_count(),
        1,
        "trapdoor sampling must use the metadata-bearing fixed path"
    );
    assert_eq!(call_counter.load(Ordering::SeqCst), measured_calls);
    gpu_device_sync();
}

/// Centered rebase is a value-preserving basis conversion, not a single-limb
/// special case.  This test keeps the source matrix on the complete two-limb
/// basis and runs both compact wire representations through the same public
/// DSL -> provider -> frozen-plan -> fixed-executor lifecycle.  BlockModSwitch
/// is included as a matrix-only sibling so the primitive inventory cannot
/// accidentally cover it only through a hand-written provider table.
#[test]
#[serial_test::serial(gpu_context)]
fn dsl_multilimb_centered_rebase_compact_and_block_switch_are_fixed_and_exact() {
    let detected = detected_gpu_device_ids();
    let Some(&device) = detected.first() else {
        panic!("CenteredRebase lifecycle requires a detected GPU");
    };
    // Run the same graph on one device.  The existing multi-GPU lifecycle
    // suite exercises the identical ownership contract when a second device
    // is present; this test intentionally keeps the numeric oracle compact.
    let devices = vec![device];
    let parameters = GpuDCRTPolyParams::new_with_gpu(
        8,
        vec![131_009, 130_817],
        8,
        devices.clone(),
        None,
        None,
        None,
    );
    let block_parameters = GpuDCRTPolyParams::new_with_gpu(
        8,
        vec![130_817],
        8,
        devices.clone(),
        None,
        Some(&parameters),
        None,
    );
    let modulus = BigInt::from_biguint(Sign::Plus, parameters.modulus().as_ref().clone());
    let digits = parameters.crt_bits().div_ceil(parameters.base_bits() as usize);
    let ring = Ring::new(modulus.clone(), parameters.ring_dimension() as usize);
    let matrix = ring.input("matrix", (1, 3));
    let small = ring.small_matrix_input("small", (digits, 3), 255u64);
    let preimage = ring.preimage_input("preimage", (digits, 3), 255u64);
    let preimage_rebased = preimage.clone().centered_rebase(modulus.clone());
    let preimage_product = ring.identity(digits).mul_small_rhs(preimage_rebased.clone());
    let graph = DslContext::new("vertical-multilimb-centered-rebase")
        .output("matrix", matrix.clone().centered_rebase(modulus.clone()))
        .expect("matrix centered rebase output")
        .output("small", small.clone().centered_rebase(modulus.clone()))
        .expect("small centered rebase output")
        .output("preimage", preimage_rebased)
        .expect("preimage centered rebase output")
        .output("preimage-product", preimage_product)
        .expect("preimage centered rebase product output")
        .output("block", matrix.block_mod_switch(130_817u64, vec![131_009, 130_817], 3u64))
        .expect("block modulus switch output")
        .build()
        .expect("DSL build")
        .validate(&ParamEnv::default())
        .expect("graph validation");

    let full_type = ConcreteMatrixType {
        modulus: modulus.clone(),
        ring_dimension: parameters.ring_dimension() as usize,
        rows: 1,
        columns: 3,
    };
    let block_type = ConcreteMatrixType { modulus: BigInt::from(130_817u64), ..full_type.clone() };
    let cpu_parameters = DCRTPolyParams::new(
        parameters.ring_dimension(),
        2,
        parameters.crt_bits(),
        parameters.base_bits(),
        Some(vec![131_009, 130_817]),
        None,
    );
    let cpu_block_parameters = DCRTPolyParams::new(
        parameters.ring_dimension(),
        1,
        parameters.crt_bits(),
        parameters.base_bits(),
        Some(vec![130_817]),
        None,
    );
    let mut cpu = cpu_backend([cpu_parameters, cpu_block_parameters]);
    let cpu_input = cpu
        .constant_matrix(
            &full_type,
            &ConstantMatrix::UnitRow { index: 0.into() },
            &ParamEnv::default(),
        )
        .expect("CPU oracle input");
    let expected_centered =
        cpu.centered_rebase(&cpu_input, &full_type).expect("CPU multi-limb centered rebase");
    let expected_block = cpu
        .block_mod_switch(&cpu_input, &block_type, &[131_009, 130_817], &BigInt::from(3u8))
        .expect("CPU BigInt block modulus switch");
    let compact_schema = ConcreteBoundedMatrixSchema {
        matrix: ConcreteMatrixType { rows: digits, ..full_type.clone() },
        max_coefficient_bound: BigInt::from(255u16),
    };
    let mut backend =
        gpu_backend_on([parameters.clone(), block_parameters.clone()], devices.clone());
    backend.select_operation([0x36; 32]).expect("select GPU operation");
    let matrix_value = backend
        .constant_matrix(
            &full_type,
            &ConstantMatrix::UnitRow { index: 0.into() },
            &ParamEnv::default(),
        )
        .expect("GPU matrix input");
    let small_value = backend
        .sample_hash_small_decomposed(
            &compact_schema.matrix,
            [0x37; 32],
            b"centered-rebase-compact-oracle",
            &BigInt::from(256u16),
            digits,
        )
        .expect("GPU small-matrix input");
    let generic_compact_bytes = backend
        .small_matrix_to_bytes(&small_value, &compact_schema, SmallMatrixSemanticKind::Generic)
        .expect("GPU generic compact encoding");
    let preimage_compact_bytes = backend
        .small_matrix_to_bytes(&small_value, &compact_schema, SmallMatrixSemanticKind::Preimage)
        .expect("GPU preimage compact encoding");
    let cpu_preimage = cpu
        .small_matrix_from_bytes(
            &compact_schema,
            &preimage_compact_bytes,
            SmallMatrixSemanticKind::Preimage,
        )
        .expect("CPU preimage codec oracle");
    let product_lhs_type =
        ConcreteMatrixType { rows: digits, columns: digits, ..full_type.clone() };
    let product_lhs = cpu
        .constant_matrix(&product_lhs_type, &ConstantMatrix::Identity, &ParamEnv::default())
        .expect("CPU compact product lhs oracle");
    let expected_preimage_product =
        cpu.multiply_small_rhs(&product_lhs, &cpu_preimage).expect("CPU compact product oracle");
    let preimage_value = small_value.clone();
    let inputs = BTreeMap::from([
        ("matrix".to_owned(), RuntimeValue::matrix(matrix_value)),
        ("small".to_owned(), RuntimeValue::small_matrix(small_value)),
        ("preimage".to_owned(), RuntimeValue::preimage(preimage_value)),
    ]);
    let RuntimeValue::Preimage(input_preimage) = inputs.get("preimage").expect("preimage input")
    else {
        panic!("preimage input changed its strict wire kind");
    };
    assert_eq!(
        backend
            .small_matrix_to_bytes(
                input_preimage,
                &compact_schema,
                SmallMatrixSemanticKind::Preimage
            )
            .expect("strict preimage input metadata"),
        preimage_compact_bytes,
        "preimage input must retain its bounded witness metadata"
    );

    let mut warmup_config = config(&graph, &parameters, device);
    warmup_config.contract.logical_to_physical_devices =
        devices.iter().map(|id| *id as usize).collect();
    warmup_config.contract.device_budgets = devices
        .iter()
        .enumerate()
        .map(|(index, device)| GpuDeviceBudget {
            device: index,
            device_bytes: gpu_memory_info(*device).expect("query GPU memory").total as u64,
            pinned_host_bytes: u64::MAX,
            host_bytes: u64::MAX,
        })
        .collect();
    warmup_config.default_tile_widths = vec![2];
    warmup_config.default_cost = vec![GpuStageCostModel::default(); devices.len()];
    for layout in &mut warmup_config.layouts {
        layout.instance_device_stride = 1;
    }
    let harness = MeasurementHarnessConfig {
        warm_up_iterations: 0,
        measured_iterations: 1,
        memory_poll_interval: Duration::ZERO,
    };
    let mut provider = GpuNodeMeasurementBackend::new(
        devices
            .iter()
            .map(|id| {
                (
                    gpu_backend_on([parameters.clone(), block_parameters.clone()], devices.clone()),
                    *id,
                )
            })
            .collect(),
        harness,
    );
    let mut warmup = mxx_runtime::gpu_warmup::warmup_gpu_from_validated_with_provider(
        &graph,
        &warmup_config,
        &mut provider,
    )
    .expect("multi-limb compact warmup must complete");
    let records = provider.warmup_dispatch_records();
    for domain in
        [CanonicalWarmupProfileDomain::CenteredRebase, CanonicalWarmupProfileDomain::BlockModSwitch]
    {
        assert!(
            records.iter().any(|record| {
                record.profile_domain == domain &&
                    record.measurement == WarmupMeasurementKind::GpuMeasured &&
                    record.range.start < record.range.end
            }),
            "{domain:?} was not measured by the production provider"
        );
    }
    assert!(provider.warmup_measurement_call_count() > 0);
    assert!(
        warmup.report.stages.iter().all(|stage| {
            stage.predicted_seconds.is_finite() &&
                stage.peak.iter().all(|peak| peak.total_bytes() < u64::MAX)
        }),
        "warmup must carry finite complete memory evidence"
    );
    assert!(
        warmup
            .report
            .stages
            .iter()
            .any(|stage| { stage.peak.iter().any(|peak| peak.total_bytes() > 0) }),
        "GPU stages must retain nonzero allocation evidence"
    );
    let measured_calls = provider.warmup_measurement_counter();
    let measured_count = provider.warmup_measurement_call_count();
    drop(provider);
    gpu_device_sync();

    warmup.plan.contract = backend
        .gpu_runtime_contract(&graph, &inputs)
        .expect("runtime contract query")
        .expect("GPU runtime contract");
    warmup.plan.validate().expect("frozen plan validation");
    for choice in &warmup.plan.nodes {
        if !choice.columns_per_job.iter().any(|width| *width > 0) {
            continue;
        }
        let Some(layout_id) = choice.output_layouts.first() else {
            continue;
        };
        let layout = warmup.plan.layout(*layout_id).expect("frozen output layout");
        if layout.columns < 3 {
            continue;
        }
        let schedule = layout.schedule(&choice.columns_per_job, 0).expect("frozen output schedule");
        assert!(schedule.intervals().iter().all(|interval| interval.start < interval.end));
        let ownership_schedule =
            if schedule.waves().any(|jobs| jobs.iter().any(|job| job.start > 0)) {
                schedule
            } else {
                layout
                    .schedule(&vec![2; choice.columns_per_job.len()], 0)
                    .expect("owner/tail candidate schedule")
            };
        assert!(ownership_schedule.waves().any(|jobs| jobs.iter().any(|job| job.start > 0)));
        assert!(
            ownership_schedule.waves().any(|jobs| jobs.iter().any(|job| job.end - job.start < 2))
        );
    }
    let output = execute_with_gpu_plan(
        &graph,
        &warmup.plan,
        &mut backend,
        inputs,
        &mut MemoryArtifactStore::default(),
        SamplingMode::Fresh,
    )
    .expect("fixed multi-limb centered-rebase execution");
    let RuntimeValue::Matrix(actual_matrix) = &output.outputs["matrix"] else {
        panic!("matrix centered rebase changed its wire kind");
    };
    let RuntimeValue::Matrix(actual_block) = &output.outputs["block"] else {
        panic!("block modulus switch changed its wire kind");
    };
    assert_eq!(
        actual_matrix.shards().first().expect("matrix output shard").value.to_cpu_matrix(),
        expected_centered
    );
    assert_eq!(
        actual_block.shards().first().expect("block output shard").value.to_cpu_matrix(),
        expected_block
    );
    let RuntimeValue::SmallMatrix(value) = &output.outputs["small"] else {
        panic!("small centered rebase changed its compact wire kind");
    };
    let bytes = backend
        .small_matrix_to_bytes(value, &compact_schema, SmallMatrixSemanticKind::Generic)
        .expect("fixed generic compact output encoding");
    assert_eq!(&bytes, &generic_compact_bytes, "small compact payload changed during rebase");

    let RuntimeValue::SmallMatrix(value) = &output.outputs["preimage"] else {
        panic!("preimage centered rebase did not produce a SmallMatrix");
    };
    let bytes = backend
        .small_matrix_to_bytes(value, &compact_schema, SmallMatrixSemanticKind::Generic)
        .expect("fixed centered-rebase compact output metadata");
    assert_eq!(&bytes, &generic_compact_bytes, "preimage compact payload changed during rebase");
    let RuntimeValue::Matrix(actual_preimage_product) = &output.outputs["preimage-product"] else {
        panic!("preimage centered rebase product changed its matrix wire kind");
    };
    assert_eq!(
        actual_preimage_product
            .shards()
            .first()
            .expect("preimage product output shard")
            .value
            .to_cpu_matrix(),
        expected_preimage_product,
        "fixed mul_small_rhs consumed the centered-rebase SmallMatrix with wrong values"
    );
    // The provider is setup-only.  Dropping the input map at the executor
    // boundary and synchronizing here exercises native consumer lifetime
    // events before the output owners are released.
    assert_eq!(measured_calls.load(Ordering::SeqCst), measured_count);
    drop(output);
    gpu_device_sync();
}
