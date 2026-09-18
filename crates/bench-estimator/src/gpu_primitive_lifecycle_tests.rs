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
    node::{ConcatAxis, ConstantMatrix, IndexRange},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::poly::{
    PolyParams,
    dcrt::gpu::{GpuDCRTPolyParams, detected_gpu_device_ids, gpu_device_sync},
};
use mxx_runtime::{
    Backend, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::{GpuWarmupProvenance, poly_gpu::gpu_backend_on},
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
                device_bytes: u64::MAX,
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
    let mut provider = GpuNodeMeasurementBackend::new(
        vec![(gpu_backend_on([source_parameters, parameters.clone()], [device]), device)],
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
    let mut backend = gpu_backend_on([source_parameters, parameters.clone()], [device]);
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
