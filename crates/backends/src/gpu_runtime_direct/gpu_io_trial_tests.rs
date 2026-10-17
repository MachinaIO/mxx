#![cfg(feature = "gpu")]

//! An I/O trial times a plan with its artifact reads and writes without the
//! artifacts it imports, and leaves nothing of its own in the store.

use crate::{
    GpuRuntime,
    artifact::FileArtifactStore,
    backend::poly_gpu::gpu_backend,
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
    },
};
use mxx_dsl::{DslContext, Ring, parallel};
use mxx_ir_core::{ParamEnv, artifact::ArtifactAvailability};
use std::{collections::BTreeMap, num::NonZeroUsize};

/// A producer that exports a family over four waves, and a consumer that
/// reads it by Zip over four waves, each get an I/O-inclusive estimate from
/// their first two waves. The consumer is planned before the producer runs:
/// its trial reads no member of the production. Neither trial leaves anything
/// in the store, and both plans then execute correctly.
#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn io_trial_plans_without_the_imported_artifacts_and_leaves_no_trial_data() {
    let cpu_params = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_params = GpuDCRTPolyParams::new(
        cpu_params.ring_dimension(),
        cpu_params.moduli().to_vec(),
        cpu_params.base_bits(),
        None,
    );
    let ring = Ring::from_crt_moduli(
        cpu_params.moduli().iter().copied().map(Into::into).collect(),
        cpu_params.ring_dimension(),
    );
    let directory = tempfile::tempdir().unwrap();
    let mut store = FileArtifactStore::new(directory.path()).unwrap();
    let productions = || {
        std::fs::read_dir(directory.path().join("productions")).map_or(0, |entries| entries.count())
    };
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    runtime.options_mut().max_parallel_instances = NonZeroUsize::new(2).unwrap();
    runtime.options_mut().io_trial = true;

    let member_ring = ring.clone();
    let members = parallel(8, move |index| {
        Ok(index.add(3).lift_to_constant_polynomial(member_ring.matrix_type((1, 1))))
    })
    .unwrap();
    let producer = DslContext::new("io-trial-producer")
        .transferred_output("members", members)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let nonce = [0x33; 32];
    let production = mxx_ir_core::artifact::production_id(
        mxx_ir_core::encoding::spec_hash(&producer.source, &producer.bindings).unwrap(),
        nonce,
    );
    let manifest =
        mxx_ir_core::artifact::export_validated_manifest(production.clone(), &producer).unwrap();
    let mut producer_plan =
        runtime.plan_with_store(producer, &BTreeMap::new(), &mut store).unwrap();
    assert!(producer_plan.report().io_predicted_seconds.is_some_and(|seconds| seconds > 0.0));
    // The family is exported member by member from the waves that produce it.
    assert!(producer_plan.physical_frame_for_test().export_templates.iter().all(|t| t.streamed));
    assert_eq!(productions(), 0, "the producer trial leaves no production behind");

    let artifact = ring.family_artifact_input(
        production.clone(),
        "members",
        8,
        (1, 1),
        ArtifactAvailability::Transferred,
    );
    let doubled = parallel(8, move |index| {
        let member = artifact.at(index);
        Ok(member.clone() + member)
    })
    .unwrap();
    let consumer = DslContext::new("io-trial-consumer")
        .output("doubled", doubled)
        .unwrap()
        .build()
        .unwrap()
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(production.clone(), manifest)]),
        )
        .unwrap();
    let mut consumer_plan =
        runtime.plan_with_store(consumer, &BTreeMap::new(), &mut store).unwrap();
    assert!(consumer_plan.report().io_predicted_seconds.is_some_and(|seconds| seconds > 0.0));
    assert_eq!(productions(), 0, "the consumer trial reads and leaves no production");

    let produced = runtime
        .execute_with_artifacts(&mut producer_plan, BTreeMap::new(), &mut store, nonce)
        .unwrap();
    assert_eq!(produced.production_id.as_ref(), Some(&production));
    // The streamed family keeps no resident copy; it is returned as its
    // committed artifact family.
    assert!(matches!(
        &produced["members"],
        crate::RuntimeValue::LazyArtifactFamily { production: returned, name, descriptor }
            if returned == &production && name == "members" && descriptor.family_count == Some(8)
    ));
    drop(produced);
    assert_eq!(productions(), 1);
    let result = runtime
        .execute_with_artifacts(&mut consumer_plan, BTreeMap::new(), &mut store, [0x34; 32])
        .unwrap();
    let output = result.output("doubled").expect("doubled output");
    for index in 0..8usize {
        let member = runtime.download_matrix_member_output(&output, index).unwrap();
        assert_eq!(
            member.entry(0, 0).coeffs_biguints()[0],
            num_bigint::BigUint::from(2 * (index + 3)),
            "member {index}"
        );
    }
}

/// An I/O trial runs only its first waves, yet exports a family that a later
/// loop also reads after the loop, members of its skipped waves included:
/// those are read from plan memory no operation wrote, which starts zeroed
/// (`device_buffer_starts_zeroed_after_the_pool_reuses_memory`), so they
/// encode as zero and planning succeeds. The plan then computes every member.
#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn io_trial_exports_the_members_of_skipped_waves() {
    let cpu_params = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_params = GpuDCRTPolyParams::new(
        cpu_params.ring_dimension(),
        cpu_params.moduli().to_vec(),
        cpu_params.base_bits(),
        None,
    );
    let ring = Ring::from_crt_moduli(
        cpu_params.moduli().iter().copied().map(Into::into).collect(),
        cpu_params.ring_dimension(),
    );
    let member_ring = ring.clone();
    let members = parallel(6, move |index| {
        Ok(index.add(3).lift_to_constant_polynomial(member_ring.matrix_type((1, 1))))
    })
    .unwrap();
    let consumed = members.clone();
    let sum =
        mxx_dsl::iterate(6, ring.zero((1, 1)), move |index, sum| Ok(sum + consumed.at(index)))
            .unwrap();
    let graph = DslContext::new("io-trial-skipped-waves")
        .transferred_output("members", members)
        .unwrap()
        .output("sum", sum)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store = FileArtifactStore::new(directory.path()).unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    // One lane per wave: six waves, of which the trial runs two.
    runtime.options_mut().max_parallel_instances = NonZeroUsize::new(1).unwrap();
    runtime.options_mut().io_trial = true;
    let mut plan = runtime.plan_with_store(graph, &BTreeMap::new(), &mut store).unwrap();
    assert_eq!(plan.frame.waves.len(), 6);
    assert!(plan.frame.export_templates.iter().all(|template| !template.streamed));
    assert!(plan.report().io_predicted_seconds.is_some(), "the trial ran");
    let result =
        runtime.execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, [0x5a; 32]).unwrap();
    let sum = runtime.download_matrix_output(&result.output("sum").expect("sum output")).unwrap();
    // 3 + 4 + ... + 8.
    assert_eq!(sum.entry(0, 0).coeffs_biguints()[0], num_bigint::BigUint::from(33u8));
}

/// Boundary compute is taken from the compute trial, and its export is
/// measured through storage without launching its Graph in the I/O trial.
#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_io_trial_skips_boundary_regions() {
    let dimension = std::env::var("MXX_TEST_IO_TRIAL_RING_DIMENSION")
        .map_or(32, |value| value.parse().unwrap());
    let cpu = DCRTPolyParams::new(dimension, 1, 28, 8, None, None);
    let gpu = GpuDCRTPolyParams::new(dimension, cpu.moduli().to_vec(), 8, None);
    let ring =
        Ring::from_crt_moduli(cpu.moduli().iter().copied().map(Into::into).collect(), dimension);
    let graph = DslContext::new("io-trial-boundary")
        .transferred_output("boundary", ring.zero((1, 1)))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store = FileArtifactStore::new(directory.path()).unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu])).unwrap();
    let plan = runtime.plan_with_store(graph, &BTreeMap::new(), &mut store).unwrap();
    assert_eq!(plan.compiled_launch_count(), 0);
    assert!(plan.report().io_predicted_seconds.unwrap() >= plan.report().predicted_seconds);
    assert_eq!(
        std::fs::read_dir(directory.path().join("productions"))
            .map_or(0, |entries| entries.count()),
        0
    );
}

/// Equal bodies with different data constants share one cold/steady pair;
/// shape and static device-loop counts keep distinct representatives.
#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_io_trial_reuses_structural_groups() {
    let dimension = std::env::var("MXX_TEST_IO_TRIAL_RING_DIMENSION")
        .map_or(32, |value| value.parse().unwrap());
    let cpu = DCRTPolyParams::new(dimension, 1, 28, 8, None, None);
    let gpu = GpuDCRTPolyParams::new(dimension, cpu.moduli().to_vec(), 8, None);
    let ring =
        Ring::from_crt_moduli(cpu.moduli().iter().copied().map(Into::into).collect(), dimension);
    let key = ring.bytes_input("key", 32);
    let mut context = DslContext::new("io-trial-reuse");
    for (name, constant, columns, count) in
        [("a", 3, 1, 2), ("b", 7, 1, 2), ("shape", 3, 2, 2), ("count", 3, 1, 3)]
    {
        let ring = ring.clone();
        let key = key.clone();
        let members = parallel(4, move |index| {
            let mut tag = mxx_dsl::HashTag::from(b"io-trial".as_slice());
            tag.push(index.add(constant));
            let initial = ring.hash_matrix(key.clone(), tag, (1, columns));
            mxx_dsl::iterate(count, initial, |_, carry| Ok(carry.clone() + carry))
        })
        .unwrap();
        context = context.transferred_output(name, members).unwrap();
    }
    let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu])).unwrap();
    let inputs = BTreeMap::from([(
        "key".into(),
        crate::RuntimeValue::Bytes(std::sync::Arc::from(rand::random::<[u8; 32]>())),
    )]);
    let mut plan = runtime.plan_with_fixed_geometry_for_test(graph, &inputs, 1, 1).unwrap();
    let groups = super::wave_groups(&plan.frame, &plan.graph).unwrap();
    assert_eq!(groups.len(), 4);
    let keys = groups
        .iter()
        .map(|g| super::io_group_key(&plan.frame, g.body_start, g.body_end, g.waves.len()))
        .collect::<Vec<_>>();
    assert_eq!(keys[0], keys[1]);
    assert_ne!(keys[0], keys[2]);
    assert_ne!(keys[0], keys[3]);
    let before = plan.graph.regions.iter().map(|r| r.launches).collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let mut store = FileArtifactStore::new(directory.path()).unwrap();
    assert!(runtime.io_trial(&mut plan, &inputs, &mut store).unwrap().unwrap() > 0.0);
    for (index, group) in groups.iter().enumerate() {
        for region in plan.graph.region_interval(group.body_start, group.body_end).unwrap() {
            assert_eq!(
                plan.graph.regions[region].launches - before[region],
                if index == 1 { 0 } else { 2 }
            );
        }
    }
}

#[test]
fn test_io_trial_reused_group_predicts_cold_and_steady() {
    let mut trial = super::IoTrial {
        ranges: Vec::new(),
        export_slots: Default::default(),
        outside_seconds: 0.0,
        outside_exports: 0,
        measured: BTreeMap::new(),
        representatives: BTreeMap::new(),
        reused_seconds: 0.0,
        loads: Vec::new(),
        published: None,
    };
    assert!(!trial.reuse("test", 0, 123, 4));
    trial.record(0, 4, 3.0);
    trial.record(0, 4, 2.0);
    assert!(trial.reuse("test", 10, 123, 4));
    // Each group contributes cold + three steady waves; only five seconds ran.
    assert_eq!(trial.extrapolate(5.0), 18.0);
}

#[test]
#[ignore = "requires a supported GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_io_trial_reuses_host_loops() {
    let dimension = std::env::var("MXX_TEST_IO_TRIAL_RING_DIMENSION")
        .map_or(32, |value| value.parse().unwrap());
    let cpu = DCRTPolyParams::new(dimension, 1, 28, 8, None, None);
    let gpu = GpuDCRTPolyParams::new(dimension, cpu.moduli().to_vec(), 8, None);
    let ring =
        Ring::from_crt_moduli(cpu.moduli().iter().copied().map(Into::into).collect(), dimension);
    let member_ring = ring.clone();
    let members = parallel(8, move |_| Ok(member_ring.zero((1, 1)))).unwrap();
    let producer = DslContext::new("io-trial-host-source")
        .transferred_output("members", members)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let production = mxx_ir_core::artifact::production_id(
        mxx_ir_core::encoding::spec_hash(&producer.source, &producer.bindings).unwrap(),
        rand::random(),
    );
    let manifest =
        mxx_ir_core::artifact::export_validated_manifest(production.clone(), &producer).unwrap();
    let artifact = ring.family_artifact_input(
        production.clone(),
        "members",
        8,
        (1, 1),
        ArtifactAvailability::Transferred,
    );
    let mut context = DslContext::new("io-trial-host-reuse");
    for (name, offset, count) in [("a", 0, 4), ("b", 1, 4), ("count", 0, 3)] {
        let sum = mxx_dsl::iterate(count, ring.zero((1, 1)), |index, sum| {
            Ok(sum + artifact.at(index.add(offset)))
        })
        .unwrap();
        context = context.output(name, sum).unwrap();
    }
    let graph = context
        .build()
        .unwrap()
        .validate_with_manifests(&ParamEnv::default(), &BTreeMap::from([(production, manifest)]))
        .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu])).unwrap();
    let mut plan =
        runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 1).unwrap();
    assert_eq!(plan.frame.external_io_loops.len(), 3);
    let bodies = plan
        .frame
        .external_io_loops
        .iter()
        .map(|b| (b.body_start, b.body_end, b.count as usize))
        .collect::<Vec<_>>();
    let keys = bodies
        .iter()
        .map(|&(start, end, count)| super::io_group_key(&plan.frame, start, end, count))
        .collect::<Vec<_>>();
    assert_eq!(keys[0], keys[1]);
    assert_ne!(keys[0], keys[2]);
    let before = plan.graph.regions.iter().map(|r| r.launches).collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let mut store = FileArtifactStore::new(directory.path()).unwrap();
    assert!(runtime.io_trial(&mut plan, &BTreeMap::new(), &mut store).unwrap().unwrap() > 0.0);
    for (index, &(start, end, _)) in bodies.iter().enumerate() {
        for region in plan.graph.region_interval(start, end).unwrap() {
            assert_eq!(
                plan.graph.regions[region].launches - before[region],
                if index == 1 { 0 } else { 2 }
            );
        }
    }
}
