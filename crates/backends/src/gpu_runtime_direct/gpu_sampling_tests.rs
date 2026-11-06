//! Probe structure, restoration, work keys and execution counts.

use super::*;
use crate::{
    backend::poly_gpu::gpu_backend,
    gpu_execution_plan::GpuImplementation,
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, GpuSignedValues, download_device_memory},
    },
};
use mxx_dsl::{DslContext, HashTag, Int, Ring, iterate};
use mxx_ir_core::{IntExpr, ParamEnv, generate_crt_basis};

fn parameter(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |value| value.parse().unwrap())
}

fn nested_plan(counts: [usize; 3]) -> (GpuRuntime, GpuExecutionPlan) {
    let dimension = parameter("MXX_TEST_SAMPLING_RING_DIMENSION", 1024) as u32;
    let columns = parameter("MXX_TEST_SAMPLING_COLUMNS", 128);
    let moduli = generate_crt_basis(dimension, 1, 28).unwrap();
    let parameters = GpuDCRTPolyParams::new(dimension, moduli.clone(), 8, None);
    let ring = Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), dimension);
    let key = ring.bytes_input("key", 32);
    let initial = (ring.zero((1, columns)), Int::constant(0));
    let (matrix, count) = iterate(counts[0], initial, |_, carry| {
        iterate(counts[1], carry, |_, carry| {
            iterate(counts[2], carry, |i, (matrix, count)| {
                let mut tag = HashTag::from(b"sampling-test".as_slice());
                tag.push(i);
                Ok((matrix + ring.hash_matrix(key.clone(), tag, (1, columns)), count.add(1)))
            })
        })
    })
    .unwrap();
    let validated = DslContext::new("sampling-test")
        .output("matrix", matrix)
        .unwrap()
        .output("count", count)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut runtime =
        GpuRuntime::new(gpu_backend([parameters])).unwrap().with_multi_lane_test_planning();
    let inputs = BTreeMap::from([(
        "key".into(),
        RuntimeValue::Bytes(Arc::from(rand::random::<[u8; 32]>())),
    )]);
    let mut plan =
        runtime.plan_with_fixed_geometry_for_test(validated, &inputs, columns, 1).unwrap();
    plan.graph.bind(&plan.frame).unwrap();
    (runtime, plan)
}

fn word(value: &GpuSignedValues) -> u64 {
    let mut bytes = [0; 8];
    download_device_memory(value.physical_device(), value.device_address() as u64, &mut bytes)
        .unwrap();
    u64::from_le_bytes(bytes)
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_constant_shape_huge_loop() {
    let count = parameter("MXX_TEST_SAMPLING_LARGE_COUNT", 1usize << 40);
    let dimension = parameter("MXX_TEST_SAMPLING_RING_DIMENSION", 1024) as u32;
    let moduli = generate_crt_basis(dimension, 1, 28).unwrap();
    let parameters = GpuDCRTPolyParams::new(dimension, moduli.clone(), 8, None);
    let ring = Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), dimension);
    let key = ring.bytes_input("key", 32);
    let matrix = iterate(count, ring.zero((1, 1)), |i, matrix| {
        let mut tag = HashTag::from(b"large-constant-shape".as_slice());
        tag.push(i);
        Ok(matrix + ring.hash_matrix(key.clone(), tag, (1, 1)))
    })
    .unwrap();
    let graph = DslContext::new("large-constant-shape")
        .output("matrix", matrix)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let inputs = BTreeMap::from([(
        "key".into(),
        RuntimeValue::Bytes(Arc::from(rand::random::<[u8; 32]>())),
    )]);
    let mut runtime = GpuRuntime::new(gpu_backend([parameters])).unwrap();
    // Planning samples one/two iterations on the GPU. It must not enumerate
    // the full count just to establish that every iteration has the same type.
    let plan = runtime.plan_with_fixed_geometry_for_test(graph, &inputs, 1, 1).unwrap();
    let loops = sampled_loops(&plan.frame, &plan.frame.program.operations, 0);
    assert_eq!(loops.len(), 1);
    assert_eq!(loops[0].count, count as u64);
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_sampling_three_levels() {
    let (runtime, mut plan) = nested_plan([2, 3, 4]);
    assert_eq!(plan.graph.regions.len(), 1);
    let loops = sampled_loops(&plan.frame, &plan.frame.program.operations, 0);
    assert_eq!(
        loops.iter().map(|l| (l.count, l.ancestors)).collect::<Vec<_>>(),
        vec![(2, 1.0), (3, 2.0), (4, 6.0)]
    );
    let mut probe_counts = Vec::new();
    sample_region(
        &plan.frame,
        &(0..plan.frame.control_resets.len()).collect::<Vec<_>>(),
        (0, plan.frame.program.operations.len() as u32),
        &loops,
        |_| {
            plan.graph.launch_region(&plan.frame, 0)?.wait()?;
            let outputs = returned_values(&plan.frame).unwrap();
            probe_counts
                .push(runtime.download_integer_family(&outputs["count"]).unwrap()[0].clone());
            Ok(1.0)
        },
    )
    .unwrap();
    assert_eq!(probe_counts, [1, 2, 2, 2].map(BigInt::from));
    let repetitions = parameter("MXX_TEST_SAMPLING_REPETITIONS", 5);
    for streamed in [false, true] {
        plan.graph.streamed_executables = streamed;
        let mut estimates = Vec::new();
        let mut whole = Vec::new();
        for _ in 0..repetitions {
            estimates.push(time_region(&mut plan.graph, &plan.frame, 0, None).unwrap());
            let start = Instant::now();
            let completion = plan.graph.launch_region(&plan.frame, 0).unwrap();
            plan.graph.finish_region(0, completion).unwrap();
            whole.push(start.elapsed().as_secs_f64());
        }
        estimates.sort_by(f64::total_cmp);
        whole.sort_by(f64::total_cmp);
        let (estimate, full) = (estimates[repetitions / 2], whole[repetitions / 2]);
        assert!(
            (estimate - full).abs() <= 0.1 * full,
            "streamed={streamed} estimate={estimate}, full={full}"
        );
    }
    let outputs = returned_values(&plan.frame).unwrap();
    assert_eq!(
        runtime.download_integer_family(&outputs["count"]).unwrap()[0].clone(),
        BigInt::from(24)
    );
    drop(runtime);
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_sampling_count_one_and_failure_restore() {
    let (_, plan) = nested_plan([1, 3, 1]);
    let loops = sampled_loops(&plan.frame, &plan.frame.program.operations, 0);
    let mut calls = 0;
    let estimate = sample_region(
        &plan.frame,
        &(0..plan.frame.control_resets.len()).collect::<Vec<_>>(),
        (0, plan.frame.program.operations.len() as u32),
        &loops,
        |_| {
            calls += 1;
            Ok(if calls == 1 { 10.0 } else { 9.0 })
        },
    )
    .unwrap();
    assert_eq!(calls, 2);
    // The negative delta is kept, but the estimate is bounded below by the baseline probe.
    assert_eq!(estimate, 10.0, "a region is never estimated below its baseline");
    for fail_at in 1..=2 {
        let mut calls = 0;
        let result = sample_region(
            &plan.frame,
            &(0..plan.frame.control_resets.len()).collect::<Vec<_>>(),
            (0, plan.frame.program.operations.len() as u32),
            &loops,
            |_| {
                calls += 1;
                if calls == fail_at {
                    Err(GpuRuntimeError::Execution("injected probe failure".into()))
                } else {
                    Ok(1.0)
                }
            },
        );
        assert!(result.is_err());
        for control in &plan.frame.control_resets {
            if let crate::gpu_physical_control::ControlReset::Loop { index, limit, status, count } =
                control
            {
                assert_eq!(word(index), 0);
                assert_eq!(word(limit), *count);
                assert_eq!(status.read().unwrap(), 0);
            }
        }
    }
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_sampling_concurrent_siblings() {
    let (_, mut plan) = nested_plan([2, 3, 4]);
    let operations = &plan.frame.program.operations;
    let outer = operations.iter().find(|op| op.body.is_some()).unwrap();
    let middle = outer.body.as_ref().unwrap().iter().find(|op| op.body.is_some()).unwrap();
    let inner = middle.body.as_ref().unwrap().iter().find(|op| op.body.is_some()).unwrap();
    let mut siblings = vec![outer.clone(), middle.clone(), inner.clone()];
    for op in &mut siblings {
        op.body = Some(Arc::from([]));
        op.predecessors = Box::new([]);
    }
    let unequal = sampled_loops(&plan.frame, &siblings, 0);
    assert_eq!(unequal.len(), 3);
    for control in &mut plan.frame.control_resets {
        if let crate::gpu_physical_control::ControlReset::Loop { count, .. } = control {
            *count = 3;
        }
    }
    let equal = sampled_loops(&plan.frame, &siblings, 0);
    assert_eq!(equal.len(), 1);
    assert_eq!(equal[0].limits.len(), 3);
    let mut calls = 0;
    assert_eq!(
        sample_region(
            &plan.frame,
            &(0..plan.frame.control_resets.len()).collect::<Vec<_>>(),
            (0, plan.frame.program.operations.len() as u32),
            &equal,
            |_| {
                calls += 1;
                Ok(10.0 * calls as f64)
            }
        )
        .unwrap(),
        30.0
    );
    assert_eq!(calls, 2);
    siblings[2].predecessors = Box::new([0]);
    assert_eq!(sampled_loops(&plan.frame, &siblings, 0).len(), 2);
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_region_key_integer_opcode() {
    let (_, mut plan) = nested_plan([1, 1, 1]);
    let implementation = plan
        .frame
        .program
        .implementations
        .register(GpuImplementation::integer_operation())
        .unwrap();
    let mut op = plan.frame.program.operations[0].clone();
    op.implementation = implementation;
    op.arguments = Box::new([KernelArg::U32(1), KernelArg::U32(17), KernelArg::U64(7)]);
    op.body = None;
    op.predecessors = Box::new([]);
    plan.frame.program.operations = vec![op.clone(), op].into_boxed_slice();
    let key = region_key(&plan.frame, 0, 1, false);
    assert_eq!(key, region_key(&plan.frame, 1, 2, false));
    plan.frame.program.operations[1].arguments[1] = KernelArg::U32(99);
    assert_eq!(key, region_key(&plan.frame, 1, 2, false), "binding numbers are ignored");
    plan.frame.program.operations[1].arguments[0] = KernelArg::U32(2);
    assert_ne!(key, region_key(&plan.frame, 1, 2, false));
    plan.frame.program.operations[1].arguments[0] = KernelArg::U32(1);
    plan.frame.program.operations[1].arguments[2] = KernelArg::U64(8);
    assert_ne!(key, region_key(&plan.frame, 1, 2, false), "constants are not normalized");
    let normalized = region_key(&plan.frame, 0, 1, true);
    assert_eq!(normalized, region_key(&plan.frame, 1, 2, true));
    plan.frame.program.operations[1].arguments[0] = KernelArg::U32(2);
    assert_ne!(normalized, region_key(&plan.frame, 1, 2, true));
    plan.frame.program.operations[1].arguments[0] = KernelArg::U32(1);
    for (left, right) in [
        (KernelArg::U64List(vec![1, 2].into()), KernelArg::U64List(vec![3, 4].into())),
        (KernelArg::I64(-1), KernelArg::I64(2)),
        (KernelArg::F64(1.0), KernelArg::F64(2.0)),
    ] {
        plan.frame.program.operations[0].arguments[2] = left;
        plan.frame.program.operations[1].arguments[2] = right;
        assert_eq!(region_key(&plan.frame, 0, 1, true), region_key(&plan.frame, 1, 2, true));
        assert_ne!(region_key(&plan.frame, 0, 1, false), region_key(&plan.frame, 1, 2, false));
    }
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_same_candidate_reuses_region() {
    let (runtime, mut plan) = nested_plan([2, 3, 4]);
    let length = plan.frame.program.operations.len();
    let mut operations = plan.frame.program.operations.to_vec();
    let mut repeated = operations.clone();
    for op in &mut repeated {
        for predecessor in &mut op.predecessors {
            *predecessor += length as u32;
        }
    }
    operations.extend(repeated);
    plan.frame.program.operations = operations.into_boxed_slice();
    let mut graph = DirectGraph::compile(&runtime.backend, &mut plan.frame, &[length]).unwrap();
    graph.bind(&plan.frame).unwrap();
    assert_eq!(graph.regions.len(), 2);
    assert_eq!(
        region_key(&plan.frame, 0, length, false),
        region_key(&plan.frame, length, 2 * length, false)
    );
    let (seconds, _) = run_trial_regions(&mut graph, &plan.frame, &[None, None], None).unwrap();
    assert_eq!(seconds[0], seconds[1]);
    assert_eq!(graph.regions[0].launches, 4);
    assert_eq!(graph.regions[1].launches, 0);
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_preimage_one_attempt() {
    let dimension = parameter("MXX_TEST_RETRY_RING_DIMENSION", 32) as u32;
    let moduli = generate_crt_basis(dimension, 1, 28).unwrap();
    let parameters = GpuDCRTPolyParams::new(dimension, moduli.clone(), 8, None);
    let digits = parameters.modulus_digits();
    let ring = Ring::from_crt_moduli(moduli.into_iter().map(IntExpr::from).collect(), dimension);
    let trapdoor = ring.sample_trapdoor(1, 4, 1u64 << 8, digits, 2_000_000);
    let preimage = trapdoor.sample_preimage(ring.zero((1, 1)), (digits + 2, 1));
    let graph = DslContext::new("trial-retry")
        .output("preimage", preimage)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([parameters])).unwrap();
    let mut plan =
        runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 1).unwrap();
    plan.graph.bind(&plan.frame).unwrap();
    assert!(!plan.frame.preimage_replays.is_empty());
    let loops = sampled_loops(&plan.frame, &plan.frame.program.operations, 0);
    assert_eq!(loops.len(), 1, "one affine column template");
    assert_eq!(loops[0].count, 1, "this fixture has one column tile");
    let mut calls = 0;
    sample_region(
        &plan.frame,
        &(0..plan.frame.control_resets.len()).collect::<Vec<_>>(),
        (0, plan.frame.program.operations.len() as u32),
        &loops,
        |_| {
            calls += 1;
            for region in 0..plan.graph.regions.len() {
                plan.graph.launch_region(&plan.frame, region)?.wait()?;
            }
            for replay in &plan.frame.preimage_replays {
                assert_eq!(word(&replay.limit), 1);
                assert_eq!(replay.status.read().unwrap().attempts, 1);
            }
            Ok(1.0)
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    assert!(
        sample_region(
            &plan.frame,
            &(0..plan.frame.control_resets.len()).collect::<Vec<_>>(),
            (0, plan.frame.program.operations.len() as u32),
            &loops,
            |_| Err(GpuRuntimeError::Execution("injected".into()))
        )
        .is_err()
    );
    for replay in &plan.frame.preimage_replays {
        assert_eq!(word(&replay.limit), u64::from(replay.planned_max));
        assert_eq!(replay.status.read().unwrap().attempts, 0);
        replay.limit.upload_u64(&[u64::from(replay.planned_max) + 17]).unwrap();
        replay.limit.wait_until_ready().unwrap();
    }
    // A disjoint region must not cap or restore these retry limits.
    sample_region(&plan.frame, &[], (0, 0), &[], |_| Ok(1.0)).unwrap();
    for replay in &plan.frame.preimage_replays {
        assert_eq!(word(&replay.limit), u64::from(replay.planned_max) + 17);
    }
}

#[test]
fn test_trial_io_cold_steady_and_boundaries() {
    let trial = IoTrial {
        ranges: vec![10..20],
        export_slots: BTreeSet::new(),
        outside_seconds: 7.0,
        outside_exports: 0,
        measured: BTreeMap::from([(10, (5, vec![3.0, 2.0])), (30, (1, vec![4.0]))]),
        representatives: BTreeMap::new(),
        reused_seconds: 0.0,
        loads: Vec::new(),
        published: Some((2, 5, 0.4)),
    };
    // Wall time includes the two measured waves, the count-one group and drain.
    assert!((trial.extrapolate(3.0 + 2.0 + 4.0 + 0.4) - 23.0).abs() < 1e-10);
}

/// Rebuild after explicit completion, preserving nested control/scratch and
/// rebinding new input owners. Compare with the unchanged resident GPU path.
#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_native_region_eviction_rebuild_rebinds_nested_graph() {
    use crate::poly::Poly;
    let (mut runtime, mut plan) = nested_plan([2, 3, 4]);
    let columns = parameter("MXX_TEST_SAMPLING_COLUMNS", 128);
    for _ in 0..3 {
        let inputs = BTreeMap::from([(
            "key".into(),
            RuntimeValue::Bytes(Arc::from(rand::random::<[u8; 32]>())),
        )]);
        plan.graph.streamed_executables = false;
        let result = runtime.execute(&mut plan, inputs.clone()).unwrap();
        let expected = runtime.download_matrix_output(&result.output("matrix").unwrap()).unwrap();
        let expected = (0..columns)
            .map(|column| expected.entry(0, column).coeffs_biguints())
            .collect::<Vec<_>>();
        drop(result);
        plan.graph.streamed_executables = true;
        for region in &mut plan.graph.regions {
            region.executable = None;
        }
        for _ in 0..2 {
            let result = runtime.execute(&mut plan, inputs.clone()).unwrap();
            let actual = runtime.download_matrix_output(&result.output("matrix").unwrap()).unwrap();
            let actual = (0..columns)
                .map(|column| actual.entry(0, column).coeffs_biguints())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            assert_eq!(
                runtime.download_integer_family_output(&result.output("count").unwrap()).unwrap(),
                vec![BigInt::from(24)]
            );
            drop(result);
            assert!(plan.graph.regions.iter().all(|region| region.executable.is_none()));
        }
    }
}

/// A probe must leave controls owned by other regions untouched.
#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_sampling_preserves_unrelated_controls() {
    let (_, plan) = nested_plan([2, 3, 4]);
    let mut selected = None;
    for (index, control) in plan.frame.control_resets.iter().enumerate() {
        if let crate::gpu_physical_control::ControlReset::Loop { limit, count, .. } = control {
            selected.get_or_insert(index);
            limit.upload_u64(&[count + 17]).unwrap();
            limit.wait_until_ready().unwrap();
        }
    }
    let selected = selected.expect("nested plan has loop controls");
    sample_region(&plan.frame, &[selected], (0, 0), &[], |_| Ok(1.0)).unwrap();
    for (index, control) in plan.frame.control_resets.iter().enumerate() {
        if let crate::gpu_physical_control::ControlReset::Loop { limit, count, .. } = control {
            assert_eq!(word(limit), if index == selected { *count } else { count + 17 });
        }
    }
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_cache_across_graphs_and_loop_counts() {
    let namespace = rand::random();
    let (_, mut first) = nested_plan([2, 3, 4]);
    let end = first.frame.program.operations.len();
    let first_key = region_key(&first.frame, 0, end, false);
    let loops = sampled_loops(&first.frame, &first.frame.program.operations, 0);
    let first_probes = probe_keys(&first.frame, 0, end, &loops);
    let mut measured = ProbeMeasurements::in_namespace(namespace);
    time_region(&mut first.graph, &first.frame, 0, Some(&mut measured)).unwrap();
    let expected_probes = 4 + 2 * usize::from(first.graph.streamed_executables);
    assert_eq!(measured.misses, expected_probes);
    measured.publish();
    // A distinct runtime and graph with different production repetition counts.
    let (_, mut second) = nested_plan([5, 6, 7]);
    second.graph.streamed_executables = first.graph.streamed_executables;
    let end = second.frame.program.operations.len();
    assert_ne!(first_key, region_key(&second.frame, 0, end, false));
    let loops = sampled_loops(&second.frame, &second.frame.program.operations, 0);
    assert_eq!(first_probes, probe_keys(&second.frame, 0, end, &loops));
    let mut reused = ProbeMeasurements::in_namespace(namespace);
    let launches = second.graph.regions[0].launches;
    let prediction = time_region(&mut second.graph, &second.frame, 0, Some(&mut reused)).unwrap();
    assert!(prediction.is_finite());
    assert_eq!(reused.hits, expected_probes);
    assert_eq!(reused.misses, 0);
    assert_eq!(second.graph.regions[0].launches, launches, "cached probes launch no GPU work");
    // A different launch shape must miss, even if arithmetic and loop counts match.
    second.frame.program.operations[0].grid[0] += 1;
    let loops = sampled_loops(&second.frame, &second.frame.program.operations, 0);
    assert_ne!(first_probes, probe_keys(&second.frame, 0, end, &loops));
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_cache_referenced_kernel_identity() {
    use crate::gpu_subgraph_kernel::{GpuKernelBuildIdentity, GpuSubgraphKernel};
    unsafe extern "C" fn unused(_: *const std::ffi::c_void) -> std::ffi::c_int {
        0
    }
    let (_, mut plan) = nested_plan([1, 1, 1]);
    let end = plan.frame.program.operations.len();
    let key = region_key(&plan.frame, 0, end, false);
    let kernel = GpuSubgraphKernel {
        name: "cache-identity-test".into(),
        build_identity: GpuKernelBuildIdentity::current(),
        inputs: vec![],
        outputs: vec![],
        parameters: vec![7],
        scratch_bytes: 0,
        entry: unused,
    };
    plan.frame.program.subgraph_kernels = vec![kernel.clone(), kernel].into_boxed_slice();
    assert_eq!(key, region_key(&plan.frame, 0, end, false), "unused registrations are irrelevant");
    let implementation = plan
        .frame
        .program
        .implementations
        .register(GpuImplementation::subgraph_kernel(0, 0))
        .unwrap();
    let op = &mut plan.frame.program.operations[0];
    op.implementation = implementation;
    op.arguments = Box::new([KernelArg::U32(0)]);
    op.body = None;
    let key = region_key(&plan.frame, 0, 1, false);
    plan.frame.program.operations[0].arguments[0] = KernelArg::U32(1);
    assert_eq!(key, region_key(&plan.frame, 0, 1, false), "registration indices are graph-local");
    plan.frame.program.subgraph_kernels[1].parameters[0] += 1;
    assert_ne!(key, region_key(&plan.frame, 0, 1, false), "referenced kernel parameters matter");
    plan.frame.program.subgraph_kernels[1].parameters[0] -= 1;
    plan.frame.program.subgraph_kernels[1].build_identity.kernel_revision.push_str("-changed");
    assert_ne!(key, region_key(&plan.frame, 0, 1, false), "referenced code revision matters");
}
