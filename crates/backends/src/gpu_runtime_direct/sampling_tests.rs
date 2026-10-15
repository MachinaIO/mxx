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
    let mut runtime = GpuRuntime::new(gpu_backend([parameters])).unwrap();
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
#[ignore = "requires a CUDA GPU"]
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
    sample_region(&plan.frame, &loops, || {
        plan.graph.launch_region(&plan.frame, 0)?.wait()?;
        let outputs = returned_values(&plan.frame).unwrap();
        probe_counts.push(runtime.download_integer_family(&outputs["count"]).unwrap()[0].clone());
        Ok(1.0)
    })
    .unwrap();
    assert_eq!(probe_counts, [1, 2, 2, 2].map(BigInt::from));
    let repetitions = parameter("MXX_TEST_SAMPLING_REPETITIONS", 5);
    let mut estimates = Vec::new();
    let mut whole = Vec::new();
    for _ in 0..repetitions {
        estimates.push(time_region(&mut plan.graph, &plan.frame, 0).unwrap());
        let start = Instant::now();
        plan.graph.launch_region(&plan.frame, 0).unwrap().wait().unwrap();
        whole.push(start.elapsed().as_secs_f64());
    }
    estimates.sort_by(f64::total_cmp);
    whole.sort_by(f64::total_cmp);
    let (estimate, full) = (estimates[repetitions / 2], whole[repetitions / 2]);
    assert!((estimate - full).abs() <= 0.1 * full, "estimate={estimate}, full={full}");
    let outputs = returned_values(&plan.frame).unwrap();
    assert_eq!(
        runtime.download_integer_family(&outputs["count"]).unwrap()[0].clone(),
        BigInt::from(24)
    );
    drop(runtime);
}

#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_sampling_count_one_and_failure_restore() {
    let (_, plan) = nested_plan([1, 3, 1]);
    let loops = sampled_loops(&plan.frame, &plan.frame.program.operations, 0);
    let mut calls = 0;
    let estimate = sample_region(&plan.frame, &loops, || {
        calls += 1;
        Ok(if calls == 1 { 10.0 } else { 9.0 })
    })
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(estimate, 8.0, "negative deltas must not be clamped");
    for fail_at in 1..=2 {
        let mut calls = 0;
        let result = sample_region(&plan.frame, &loops, || {
            calls += 1;
            if calls == fail_at {
                Err(GpuRuntimeError::Execution("injected probe failure".into()))
            } else {
                Ok(1.0)
            }
        });
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
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_trial_sampling_concurrent_siblings() {
    let (_, mut plan) = nested_plan([2, 3, 4]);
    let operations = &plan.frame.program.operations;
    let outer = operations.iter().find(|op| op.body.is_some()).unwrap();
    let middle = outer.body.as_ref().unwrap().iter().find(|op| op.body.is_some()).unwrap();
    let inner = middle.body.as_ref().unwrap().iter().find(|op| op.body.is_some()).unwrap();
    let mut siblings = vec![outer.clone(), middle.clone(), inner.clone()];
    for op in &mut siblings {
        op.body = Some(Box::new([]));
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
        sample_region(&plan.frame, &equal, || {
            calls += 1;
            Ok(10.0 * calls as f64)
        })
        .unwrap(),
        30.0
    );
    assert_eq!(calls, 2);
    siblings[2].predecessors = Box::new([0]);
    assert_eq!(sampled_loops(&plan.frame, &siblings, 0).len(), 2);
}

#[test]
#[ignore = "requires a CUDA GPU"]
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
#[ignore = "requires a CUDA GPU"]
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
    let (seconds, _) = run_trial_regions(&mut graph, &plan.frame, &[None, None]).unwrap();
    assert_eq!(seconds[0], seconds[1]);
    assert_eq!(graph.regions[0].launches, 4);
    assert_eq!(graph.regions[1].launches, 0);
}

#[test]
#[ignore = "requires a CUDA GPU"]
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
    assert!(loops.is_empty());
    let mut calls = 0;
    sample_region(&plan.frame, &loops, || {
        calls += 1;
        for region in 0..plan.graph.regions.len() {
            plan.graph.launch_region(&plan.frame, region)?.wait()?;
        }
        for replay in &plan.frame.preimage_replays {
            assert_eq!(word(&replay.limit), 1);
            assert_eq!(replay.status.read().unwrap().attempts, 1);
        }
        Ok(1.0)
    })
    .unwrap();
    assert_eq!(calls, 1);
    assert!(
        sample_region(&plan.frame, &loops, || Err(GpuRuntimeError::Execution("injected".into())))
            .is_err()
    );
    for replay in &plan.frame.preimage_replays {
        assert_eq!(word(&replay.limit), u64::from(replay.planned_max));
        assert_eq!(replay.status.read().unwrap().attempts, 0);
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
