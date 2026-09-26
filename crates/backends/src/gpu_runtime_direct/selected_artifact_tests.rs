#![cfg(feature = "gpu")]

//! A root device selector imports only the artifact member it actually reads.

use crate::{
    GpuRuntime, RuntimeValue,
    artifact::{ArtifactKey, ArtifactPayload, ArtifactStore, MemoryArtifactStore},
    backend::{GpuResidentValue, poly::cpu_backend, poly_gpu::gpu_backend},
    executor::{ExecutionConfig, execute_in_session},
    gpu_execution_plan::{PhysicalValue, PhysicalValueId},
    matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
    },
    sampler::trapdoor::DCRTTrapdoor,
    session::SessionStore,
};
use mxx_dsl::{DslContext, Int, IntType, Ring, parallel};
use mxx_ir_core::{
    Graph, GraphOutput, IntExpr, NodeHandle, ParamEnv, WireType,
    artifact::ArtifactAvailability,
    node::{ArtifactInput, NodeKind},
    types::ConcreteWireType,
};
use num_bigint::BigInt;
use num_traits::Zero;
use std::{collections::BTreeMap, sync::Arc};

#[test]
#[serial_test::serial]
fn root_dynamic_family_get_loads_only_selected_artifact_on_each_replay() {
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
    let producer_ring = ring.clone();
    let producer = DslContext::new("selected-root-family-producer")
        .cached_output(
            "members",
            parallel(3, move |index| Ok(producer_ring.polynomial([index.expression()? + 11])))
                .unwrap(),
        )
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let mut store = MemoryArtifactStore::default();
    let produced = execute_in_session(
        &producer,
        &mut cpu_backend([cpu_params]),
        BTreeMap::new(),
        &mut store,
        [0x51; 32],
        ExecutionConfig::default(),
    )
    .unwrap();
    let production = produced.production_id.expect("producer identity");
    let manifest = store.load_finalized_manifest(&production).unwrap();

    let consumer = DslContext::new("selected-root-family-consumer");
    let selector: Int = consumer.input("selector", IntType).unwrap();
    let family = ring.family_artifact_input(
        production.clone(),
        "members",
        3,
        (1, 1),
        ArtifactAvailability::Cached,
    );
    let consumer = consumer
        .output("selected", family.at(selector))
        .unwrap()
        .build()
        .unwrap()
        .validate_with_manifests(
            &ParamEnv::default(),
            &BTreeMap::from([(production.clone(), manifest)]),
            crate::openfhe_guard::gen_modulus_and_warmup,
        )
        .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    runtime.options_mut().integer_input_ranges.insert("selector".into(), 0.into()..=2.into());
    let bind = |index: usize| {
        BTreeMap::from([("selector".into(), RuntimeValue::Int(BigInt::from(index)))])
    };
    let mut plan = runtime.plan(consumer, &bind(2)).unwrap();
    let keys = (0..3)
        .map(|index| ArtifactKey {
            production: production.clone(),
            name: "members".into(),
            index: Some(index),
        })
        .collect::<Vec<_>>();
    assert_eq!(keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>(), [0; 3]);

    for (replay, index) in [2, 1].into_iter().enumerate() {
        let before = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        let result = runtime
            .execute_with_artifacts(&mut plan, bind(index), &mut store, [0x61 + replay as u8; 32])
            .unwrap();
        let matrix = runtime
            .download_matrix_output(&result.output("selected").expect("selected output"))
            .unwrap();
        let coefficients = matrix.entry(0, 0).coeffs_biguints();
        assert_eq!(coefficients[0], num_bigint::BigUint::from(index + 11));
        assert!(coefficients[1..].iter().all(num_bigint::BigUint::is_zero));
        let after = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        for member in 0..3 {
            assert_eq!(after[member] - before[member], usize::from(member == index));
        }
        // A consumer that only imports opens no producer session.
        assert!(result.production_id.is_none());
    }
    // Reusing a nonce is valid for a consumer, since it records no session.
    runtime.execute_with_artifacts(&mut plan, bind(0), &mut store, [0x61; 32]).unwrap();
}

#[test]
#[serial_test::serial]
fn root_dynamic_typed_blob_import_preserves_selected_length_on_replay() {
    let cpu_params = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_params = GpuDCRTPolyParams::new(
        cpu_params.ring_dimension(),
        cpu_params.moduli().to_vec(),
        cpu_params.base_bits(),
        None,
    );
    let type_name = "selected-test-blob".to_owned();
    let schema_hash = [0x39; 32];
    let element = WireType::TypedBlob { type_name: type_name.clone(), schema_hash };
    let family =
        WireType::IndexedFamily { element: Box::new(element.clone()), count: IntExpr::constant(3) };
    let producer_input = NodeHandle::new(
        NodeKind::Input { name: "family".into(), wire_type: family.clone(), artifact: None },
        vec![],
        vec![family.clone()],
    )
    .output(0)
    .unwrap();
    let producer = Graph::freeze(
        "selected-typed-blob-producer",
        vec![],
        BTreeMap::from([(
            "members".into(),
            GraphOutput { value: producer_input, availability: Some(ArtifactAvailability::Cached) },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let producer = mxx_ir_core::validate(
        &producer,
        &ParamEnv::default(),
        crate::openfhe_guard::gen_modulus_and_warmup,
    )
    .unwrap();
    let payloads = [vec![0x11], vec![0x22, 0x23, 0x24], vec![0x31, 0x32, 0x33, 0x34, 0x35]];
    let members = payloads
        .iter()
        .map(|bytes| RuntimeValue::TypedBlob {
            type_name: type_name.clone(),
            schema_hash,
            bytes: bytes.clone().into(),
        })
        .collect::<Vec<_>>();
    let mut store = MemoryArtifactStore::default();
    let produced = execute_in_session(
        &producer,
        &mut cpu_backend([cpu_params]),
        BTreeMap::from([(
            "family".into(),
            RuntimeValue::IndexedFamily {
                element_type: ConcreteWireType::TypedBlob {
                    type_name: type_name.clone(),
                    schema_hash,
                },
                values: members.into(),
            },
        )]),
        &mut store,
        [0x71; 32],
        ExecutionConfig::default(),
    )
    .unwrap();
    let production = produced.production_id.expect("producer identity");
    let manifest = store.load_finalized_manifest(&production).unwrap();

    let artifact = NodeHandle::new(
        NodeKind::Input {
            name: "members".into(),
            wire_type: family.clone(),
            artifact: Some(ArtifactInput {
                production_id: production.clone(),
                artifact_name: "members".into(),
                availability: ArtifactAvailability::Cached,
            }),
        },
        vec![],
        vec![family],
    )
    .output(0)
    .unwrap();
    let selector = NodeHandle::new(
        NodeKind::Input { name: "selector".into(), wire_type: WireType::Int, artifact: None },
        vec![],
        vec![WireType::Int],
    )
    .output(0)
    .unwrap();
    let selected =
        NodeHandle::new(NodeKind::FamilyGetDynamic, vec![artifact, selector], vec![element])
            .output(0)
            .unwrap();
    let consumer = Graph::freeze(
        "selected-typed-blob-consumer",
        vec![],
        BTreeMap::from([("selected".into(), GraphOutput { value: selected, availability: None })]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let consumer = mxx_ir_core::validate_with_manifests(
        &consumer,
        &ParamEnv::default(),
        &BTreeMap::from([(production.clone(), manifest)]),
        crate::openfhe_guard::gen_modulus_and_warmup,
    )
    .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    runtime.options_mut().integer_input_ranges.insert("selector".into(), 0.into()..=2.into());
    let bind = |index: usize| {
        BTreeMap::from([("selector".into(), RuntimeValue::Int(BigInt::from(index)))])
    };
    let mut plan = runtime.plan_with_store(consumer, &bind(2), &mut store).unwrap();
    let keys = (0..3)
        .map(|index| ArtifactKey {
            production: production.clone(),
            name: "members".into(),
            index: Some(index),
        })
        .collect::<Vec<_>>();
    assert_eq!(keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>(), [0; 3]);

    for (replay, index) in [2, 0, 1].into_iter().enumerate() {
        let before = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        let result = runtime
            .execute_with_artifacts(&mut plan, bind(index), &mut store, [0x81 + replay as u8; 32])
            .unwrap();
        assert_eq!(
            runtime
                .download_bytes_output(&result.output("selected").expect("selected output"))
                .unwrap(),
            payloads[index],
        );
        let after = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        for member in 0..3 {
            assert_eq!(after[member] - before[member], usize::from(member == index));
        }
    }
}

#[test]
#[serial_test::serial]
fn root_dynamic_integer_import_replays_mixed_signed_widths() {
    let cpu_params = DCRTPolyParams::new(8, 2, 20, 4, None, None);
    let gpu_params = GpuDCRTPolyParams::new(
        cpu_params.ring_dimension(),
        cpu_params.moduli().to_vec(),
        cpu_params.base_bits(),
        None,
    );
    let family =
        WireType::IndexedFamily { element: Box::new(WireType::Int), count: IntExpr::constant(3) };
    let producer_input = NodeHandle::new(
        NodeKind::Input { name: "family".into(), wire_type: family.clone(), artifact: None },
        vec![],
        vec![family.clone()],
    )
    .output(0)
    .unwrap();
    let producer = Graph::freeze(
        "selected-integer-producer",
        vec![],
        BTreeMap::from([(
            "members".into(),
            GraphOutput { value: producer_input, availability: Some(ArtifactAvailability::Cached) },
        )]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let producer = mxx_ir_core::validate(
        &producer,
        &ParamEnv::default(),
        crate::openfhe_guard::gen_modulus_and_warmup,
    )
    .unwrap();
    let values = [
        -(BigInt::from(1u8) << 80usize) + BigInt::from(7u8),
        BigInt::from(42u8),
        (BigInt::from(1u8) << 127usize) + BigInt::from(3u8),
    ];
    let mut store = MemoryArtifactStore::default();
    let produced = execute_in_session(
        &producer,
        &mut cpu_backend([cpu_params]),
        BTreeMap::from([(
            "family".into(),
            RuntimeValue::IndexedFamily {
                element_type: ConcreteWireType::Int,
                values: values.iter().cloned().map(RuntimeValue::Int).collect::<Vec<_>>().into(),
            },
        )]),
        &mut store,
        [0x91; 32],
        ExecutionConfig::default(),
    )
    .unwrap();
    let production = produced.production_id.expect("producer identity");
    let manifest = store.load_finalized_manifest(&production).unwrap();

    let artifact = NodeHandle::new(
        NodeKind::Input {
            name: "members".into(),
            wire_type: family.clone(),
            artifact: Some(ArtifactInput {
                production_id: production.clone(),
                artifact_name: "members".into(),
                availability: ArtifactAvailability::Cached,
            }),
        },
        vec![],
        vec![family],
    )
    .output(0)
    .unwrap();
    let selector = NodeHandle::new(
        NodeKind::Input { name: "selector".into(), wire_type: WireType::Int, artifact: None },
        vec![],
        vec![WireType::Int],
    )
    .output(0)
    .unwrap();
    let selected =
        NodeHandle::new(NodeKind::FamilyGetDynamic, vec![artifact, selector], vec![WireType::Int])
            .output(0)
            .unwrap();
    let consumer = Graph::freeze(
        "selected-integer-consumer",
        vec![],
        BTreeMap::from([("selected".into(), GraphOutput { value: selected, availability: None })]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let consumer = mxx_ir_core::validate_with_manifests(
        &consumer,
        &ParamEnv::default(),
        &BTreeMap::from([(production.clone(), manifest)]),
        crate::openfhe_guard::gen_modulus_and_warmup,
    )
    .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    runtime.options_mut().integer_input_ranges.insert("selector".into(), 0.into()..=2.into());
    let bind = |index: usize| {
        BTreeMap::from([("selector".into(), RuntimeValue::Int(BigInt::from(index)))])
    };
    let mut plan = runtime.plan_with_store(consumer, &bind(2), &mut store).unwrap();
    let keys = (0..3)
        .map(|index| ArtifactKey {
            production: production.clone(),
            name: "members".into(),
            index: Some(index),
        })
        .collect::<Vec<_>>();
    assert_eq!(keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>(), [0; 3]);

    for (replay, index) in [2, 0, 1].into_iter().enumerate() {
        let before = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        let result = runtime
            .execute_with_artifacts(&mut plan, bind(index), &mut store, [0xa1 + replay as u8; 32])
            .unwrap();
        assert_eq!(
            runtime
                .download_integer_family_output(
                    &result.output("selected").expect("selected output")
                )
                .unwrap(),
            vec![values[index].clone()],
        );
        let after = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        for member in 0..3 {
            assert_eq!(after[member] - before[member], usize::from(member == index));
        }
    }
}

#[test]
#[serial_test::serial]
fn root_dynamic_trapdoor_import_preserves_public_and_six_secret_leaves() {
    let cpu_params = DCRTPolyParams::new(32, 1, 28, 8, None, None);
    let gpu_params = GpuDCRTPolyParams::new(
        cpu_params.ring_dimension(),
        cpu_params.moduli().to_vec(),
        cpu_params.base_bits(),
        None,
    );
    let digits = cpu_params.modulus_digits();
    let oracle_params = cpu_params.clone();
    let ring = Ring::from_crt_moduli(
        cpu_params.moduli().iter().copied().map(Into::into).collect(),
        cpu_params.ring_dimension(),
    );
    let producer_ring = ring.clone();
    let trapdoors =
        parallel(2, move |_| Ok(producer_ring.sample_trapdoor(1, 4, 1u64 << 8, digits, 1_000_000)))
            .unwrap();
    let producer = DslContext::new("selected-trapdoor-producer")
        .transferred_trapdoor_family_output("secrets", trapdoors)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let mut store = MemoryArtifactStore::default();
    let produced = execute_in_session(
        &producer,
        &mut cpu_backend([cpu_params]),
        BTreeMap::new(),
        &mut store,
        rand::random(),
        ExecutionConfig::default(),
    )
    .unwrap();
    let production = produced.production_id.expect("producer identity");
    let manifest = store.load_finalized_manifest(&production).unwrap();
    let descriptor = manifest.artifacts["secrets"].clone();

    let trapdoor_schema = ring.sample_trapdoor(1, 4, 1u64 << 8, digits, 1_000_000);
    let element = trapdoor_schema.value_handle().node().output_types()[1].clone();
    let family =
        WireType::IndexedFamily { element: Box::new(element.clone()), count: IntExpr::constant(2) };
    let artifact = NodeHandle::new(
        NodeKind::Input {
            name: "secrets".into(),
            wire_type: family.clone(),
            artifact: Some(ArtifactInput {
                production_id: production.clone(),
                artifact_name: "secrets".into(),
                availability: ArtifactAvailability::Transferred,
            }),
        },
        vec![],
        vec![family],
    )
    .output(0)
    .unwrap();
    let selector = NodeHandle::new(
        NodeKind::Input { name: "selector".into(), wire_type: WireType::Int, artifact: None },
        vec![],
        vec![WireType::Int],
    )
    .output(0)
    .unwrap();
    let selected =
        NodeHandle::new(NodeKind::FamilyGetDynamic, vec![artifact, selector], vec![element])
            .output(0)
            .unwrap();
    let consumer = Graph::freeze(
        "selected-trapdoor-consumer",
        vec![],
        BTreeMap::from([("selected".into(), GraphOutput { value: selected, availability: None })]),
        vec![],
        vec![],
        BTreeMap::new(),
    )
    .unwrap()
    .0;
    let consumer = mxx_ir_core::validate_with_manifests(
        &consumer,
        &ParamEnv::default(),
        &BTreeMap::from([(production.clone(), manifest)]),
        crate::openfhe_guard::gen_modulus_and_warmup,
    )
    .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    runtime.options_mut().integer_input_ranges.insert("selector".into(), 0.into()..=1.into());
    let bind = |index: usize| {
        BTreeMap::from([("selector".into(), RuntimeValue::Int(BigInt::from(index)))])
    };
    let mut plan = runtime.plan(consumer, &bind(1)).unwrap();
    let keys = (0..2)
        .map(|index| ArtifactKey {
            production: production.clone(),
            name: "secrets".into(),
            index: Some(index),
        })
        .collect::<Vec<_>>();
    assert_eq!(keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>(), [0; 2]);

    for (replay, index) in [1, 0].into_iter().enumerate() {
        let before = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        {
            let result = runtime
                .execute_with_artifacts(&mut plan, bind(index), &mut store, rand::random())
                .unwrap();
            assert!(result.output("selected").is_some(), "selected trapdoor output");
        }
        let frame = plan.physical_frame_for_test();
        let returned_id = frame.output_ids["selected"];
        let secret = &frame.owners[&returned_id];
        assert_eq!(secret.physical().encodings.len(), 6);
        let imported_id = *frame
            .trapdoor_public_ids
            .keys()
            .find(|id| **id != returned_id && frame.owners[*id].physical().encodings.len() == 6)
            .expect("selected imported secret with paired public matrix");
        let public_id = frame.trapdoor_public_ids[&imported_id];
        let public = runtime
            .download_matrix(&RuntimeValue::Resident(frame.owners[&public_id].clone()))
            .unwrap();
        let after = keys.iter().map(|key| store.load_count(key)).collect::<Vec<_>>();
        for member in 0..2 {
            assert_eq!(after[member] - before[member], usize::from(member == index));
        }
        let ArtifactPayload::Trapdoor { public_bytes, secret_bytes } =
            store.load(&keys[index], &descriptor).unwrap()
        else {
            panic!("selected CPU oracle artifact must be a trapdoor");
        };
        let expected_public = DCRTPolyMatrix::try_from_eval_artifact(
            &oracle_params,
            public.size().0,
            public.size().1,
            &public_bytes,
        )
        .unwrap();
        let expected_secret =
            DCRTTrapdoor::try_from_compact_bytes(&oracle_params, &secret_bytes).unwrap();
        assert_eq!(public.to_compact_bytes(), expected_public.to_compact_bytes());
        let expected_leaves = [
            expected_secret.r_cpu(),
            expected_secret.e_cpu(),
            expected_secret.a_mat_cpu(),
            expected_secret.b_mat_cpu(),
            expected_secret.d_mat_cpu(),
            expected_secret.re_cpu(),
        ];
        for (leaf, expected_leaf) in expected_leaves.iter().enumerate() {
            let actual = download_trapdoor_leaf(&runtime, frame, imported_id, leaf);
            assert_eq!(
                actual.to_compact_bytes(),
                expected_leaf.to_compact_bytes(),
                "imported trapdoor leaf {leaf} differs on replay {replay}",
            );
            let returned = download_resident_trapdoor_leaf(&runtime, secret, leaf);
            assert_eq!(
                returned.to_compact_bytes(),
                expected_leaf.to_compact_bytes(),
                "returned trapdoor leaf {leaf} differs on replay {replay}",
            );
        }
    }
}

fn download_resident_trapdoor_leaf(
    runtime: &GpuRuntime,
    secret: &Arc<GpuResidentValue>,
    leaf: usize,
) -> DCRTPolyMatrix {
    let physical = secret.physical();
    let ty = crate::gpu_physical_lowering::trapdoor_leaf_types(&physical.ty).unwrap()[leaf].clone();
    let parts = physical
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
        encodings: Box::new([physical.encodings[leaf].clone()]),
        parts: parts.into_boxed_slice(),
        integer_ranges: BTreeMap::new(),
    });
    let owner = Arc::new(secret.with_physical_view(view).unwrap());
    runtime.download_matrix(&RuntimeValue::Resident(owner)).unwrap()
}

fn download_trapdoor_leaf(
    runtime: &GpuRuntime,
    frame: &crate::gpu_physical_lowering::PhysicalFrame,
    secret_id: PhysicalValueId,
    leaf: usize,
) -> DCRTPolyMatrix {
    let secret = &frame.owners[&secret_id];
    let part = secret
        .physical()
        .parts
        .iter()
        .find(|part| part.leaf as usize == leaf)
        .expect("trapdoor leaf part");
    let owner = frame
        .owners
        .values()
        .find(|candidate| {
            candidate.physical().ty.matrix_type().is_some() &&
                candidate.physical().parts.iter().any(|other| other.storage == part.storage)
        })
        .expect("trapdoor leaf matrix owner");
    runtime.download_matrix(&RuntimeValue::Resident(owner.clone())).unwrap()
}

/// Run a parallel-loop family consumed by a later sequential loop; with
/// `publish`, the family is also a transferred artifact.
fn run_family_consumed_by_later_loop(publish: bool) {
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
    // Members are runtime values of the loop index, not static constants.
    let members = parallel(4, move |index| {
        Ok(index.add(3).lift_to_constant_polynomial(member_ring.matrix_type((1, 1))))
    })
    .unwrap();
    let consumed = members.clone();
    let sum =
        mxx_dsl::iterate(4, ring.zero((1, 1)), move |index, sum| Ok(sum + consumed.at(index)))
            .unwrap();
    let context = DslContext::new("family-consumed-by-later-loop");
    let context =
        if publish { context.transferred_output("members", members).unwrap() } else { context };
    let graph = context
        .output("sum", sum)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
        .unwrap();
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
    let mut plan = runtime.plan(graph, &BTreeMap::new()).unwrap();
    let mut store = MemoryArtifactStore::default();
    let result =
        runtime.execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, [0x71; 32]).unwrap();
    let sum = runtime.download_matrix_output(&result.output("sum").expect("sum output")).unwrap();
    assert_eq!(sum.entry(0, 0).coeffs_biguints()[0], num_bigint::BigUint::from(18u8));
    if publish {
        let production = result.production_id.clone().expect("producer identity");
        let handles = result.artifact_handles.clone();
        drop(result);
        let manifest = store.load_finalized_manifest(&production).unwrap();
        assert_eq!(manifest.artifacts.len(), 1);
        // The same nonce replays the finalized production: the outputs are
        // recomputed and the committed artifacts are returned, not rewritten.
        let replay = runtime
            .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, [0x71; 32])
            .unwrap();
        assert_eq!(replay.production_id.as_ref(), Some(&production));
        assert_eq!(replay.artifact_handles, handles);
        let sum =
            runtime.download_matrix_output(&replay.output("sum").expect("sum output")).unwrap();
        assert_eq!(sum.entry(0, 0).coeffs_biguints()[0], num_bigint::BigUint::from(18u8));
        assert_eq!(store.load_finalized_manifest(&production).unwrap(), manifest);
    }
}

/// Every member of a published family is committed even though a later loop
/// also consumes it, and the consumer reads every member.
#[test]
#[serial_test::serial]
fn exported_family_consumed_by_later_loop_commits_every_member() {
    run_family_consumed_by_later_loop(true);
}

/// A value live across the family's replayed region is freed by the host
/// after its last region, which also holds the consumer's device loop.
#[test]
#[serial_test::serial]
fn family_consumed_by_later_loop_reads_every_member() {
    run_family_consumed_by_later_loop(false);
}
