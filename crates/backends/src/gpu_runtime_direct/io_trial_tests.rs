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
#[serial_test::serial]
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
    runtime.options_mut().io_trial_waves = NonZeroUsize::new(2);

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
