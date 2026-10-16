#![cfg(feature = "gpu")]

//! Wave-loop outputs beyond plain matrix families: trapdoor families read by member, families of
//! families exported flattened (by the root lanes or by their nested loops), unread and sliced
//! nested outputs, resident families read at a runtime base, and loop-dependent scalar powers and
//! bit positions. Every test compares the GPU with the CPU on fixed-geometry plans of one lane,
//! so that every loop replays later waves.

use crate::{
    GpuRuntime, RuntimeValue,
    artifact::MemoryArtifactStore,
    backend::{poly::cpu_backend, poly_gpu::gpu_backend_on},
    executor::{ExecutionConfig, execute, execute_in_session},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
            params::DCRTPolyParams,
        },
    },
    session::SessionStore,
    transcript::SamplingMode,
};
use mxx_dsl::{DslContext, Family, Int, IntType, Mat, Ring, parallel};
use mxx_ir_core::{
    IntExpr, ParamEnv, ValidatedGraph,
    artifact::ArtifactAvailability,
    node::{ConcatAxis, ConstantMatrix, IndexRange},
};
use num_bigint::BigInt;
use num_traits::Zero;
use std::collections::BTreeMap;

fn parameters() -> (DCRTPolyParams, GpuDCRTPolyParams, Ring) {
    let cpu = DCRTPolyParams::new(32, 2, 40, 20, None, None);
    let gpu =
        GpuDCRTPolyParams::new(cpu.ring_dimension(), cpu.moduli().to_vec(), cpu.base_bits(), None);
    let ring = Ring::from_crt_moduli(
        cpu.moduli().iter().copied().map(Into::into).collect(),
        cpu.ring_dimension(),
    );
    (cpu, gpu, ring)
}

fn runtime(gpu: GpuDCRTPolyParams) -> GpuRuntime {
    GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap()
}

/// The GPU's outputs `names` of `graph` (one lane, two waves of replay) equal the CPU's, on two
/// executions of one plan.
fn assert_matches_cpu(
    graph: ValidatedGraph,
    inputs: BTreeMap<String, RuntimeValue>,
    names: &[String],
    (cpu, gpu): (DCRTPolyParams, GpuDCRTPolyParams),
) {
    let mut store = MemoryArtifactStore::default();
    let expected = execute(
        &graph,
        &mut cpu_backend([cpu]),
        inputs.clone(),
        &mut store,
        SamplingMode::Fresh,
        ExecutionConfig::default(),
    )
    .unwrap();
    let mut runtime = runtime(gpu);
    let mut plan = runtime.plan_with_fixed_geometry_for_test(graph, &inputs, 1, 1).unwrap();
    for replay in 0..2 {
        let result = runtime
            .execute_with_artifacts(&mut plan, inputs.clone(), &mut store, rand::random())
            .unwrap();
        for name in names {
            let actual = runtime.download_matrix_output(&result.output(name).unwrap()).unwrap();
            let RuntimeValue::Matrix(expected) = &expected.outputs[name] else {
                panic!("CPU output {name} is a matrix")
            };
            assert_eq!(&actual, expected.as_cpu_full().unwrap(), "{name}, replay {replay}");
        }
    }
}

/// A parallel loop samples one trapdoor per member and another loop reads them by a computed
/// member index: the family stays resident with its public matrices, later waves' trapdoors in
/// the member arena, and every preimage satisfies `B K = target`.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_parallel_trapdoor_family_feeds_member_preimages() {
    let (cpu, gpu, ring) = parameters();
    let digits = cpu.modulus_digits();
    let base = 1u32 << cpu.base_bits();
    let bound = BigInt::from(
        crate::sampler::bounds::default_preimage_cutoff(
            cpu.ring_dimension(),
            2,
            digits,
            base,
            4.0,
            mxx_ir_core::types::TrapdoorInstantiation::Computational,
        )
        .unwrap(),
    );
    let (trapdoors, per) = (3usize, 2usize);
    let anchors = {
        let ring = ring.clone();
        parallel(trapdoors, move |_| Ok(ring.sample_trapdoor(2, 4, base, digits, bound.clone())))
            .unwrap()
    };
    let residuals = {
        let ring = ring.clone();
        parallel(trapdoors * per, move |member| {
            let owner = member.div(per);
            let target = ring.uniform_residue((2, 1));
            let preimage =
                anchors.at(owner.clone()).sample_preimage(target.clone(), (2 * (digits + 2), 1));
            Ok(anchors.public_matrices().at(owner).mul_small_rhs(preimage) - target)
        })
        .unwrap()
    };
    let mut context = DslContext::new("trapdoor-family-preimages");
    for member in 0..trapdoors * per {
        context = context.output(format!("residual_{member}"), residuals.at(member)).unwrap();
    }
    let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let mut runtime = runtime(gpu);
    let mut plan =
        runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 1).unwrap();
    let mut store = MemoryArtifactStore::default();
    for replay in 0..2 {
        let result = runtime
            .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
            .unwrap();
        for member in 0..trapdoors * per {
            let residual = runtime
                .download_matrix_output(&result.output(&format!("residual_{member}")).unwrap())
                .unwrap();
            assert!(
                (0..residual.size().0).all(|row| residual
                    .entry(row, 0)
                    .coeffs_biguints()
                    .iter()
                    .all(Zero::is_zero)),
                "replay {replay}, member {member}: B K = target"
            );
        }
    }
}

/// A family of families published flattened (`parallel(n k, |m| f.at(m / k).at(m % k))`), once
/// from a nested loop of each lane and once from a packed family of each lane: every member
/// reaches its flat index, and the flattening loop itself never runs.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_flattened_family_exports_every_member() {
    let (cpu, gpu, ring) = parameters();
    let (outer, inner) = (3usize, 4usize);
    let value = |i: &Int, j: IntExpr| -> Result<IntExpr, mxx_dsl::DslError> {
        Ok(IntExpr::Add(
            Box::new(IntExpr::Mul(Box::new(i.expression()?), Box::new(IntExpr::constant(16)))),
            Box::new(j),
        ))
    };
    let instances = {
        let ring = ring.clone();
        parallel(outer, move |i| {
            let looped = parallel(inner, |j| {
                Ok(ring.polynomial([value(&i, j.expression()?)? + IntExpr::constant(1)]))
            })?;
            let packed = Family::pack(vec![
                ring.polynomial([value(&i, IntExpr::constant(100))?]),
                ring.polynomial([value(&i, IntExpr::constant(200))?]),
            ])?;
            Ok((looped, packed))
        })
        .unwrap()
    };
    let flatten = |family: Family<Family<Mat>>, per: usize| {
        let per_int = Int::evaluate(IntExpr::constant(per));
        parallel(outer * per, move |member| {
            Ok(family.at(member.clone().div(per_int.clone())).at(member.rem(per_int.clone())))
        })
        .unwrap()
    };
    let looped = flatten(instances.field(|(looped, _)| looped.clone()).unwrap(), inner);
    let packed = flatten(instances.field(|(_, packed)| packed.clone()).unwrap(), 2);
    let producer = DslContext::new("flattened-families")
        .cached_output("looped", looped)
        .unwrap()
        .cached_output("packed", packed)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut store = MemoryArtifactStore::default();
    let mut runtime = runtime(gpu.clone());
    let mut plan =
        runtime.plan_with_fixed_geometry_for_test(producer, &BTreeMap::new(), 1, 1).unwrap();
    // The nested loop exports the looped members at its lane's block; the lanes export the
    // packed ones.
    let templates = &plan.physical_frame_for_test().export_templates;
    assert!(templates.iter().any(|t| t.name == "looped" && t.parent_stride == Some(inner)));
    assert!(
        templates.iter().any(|t| t.name == "packed" && t.streamed && t.parent_stride.is_none())
    );
    let result = runtime
        .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
        .unwrap();
    let production = result.production_id.clone().expect("a producing execution");
    let manifest = store.load_finalized_manifest(&production).unwrap();
    let mut consumer = DslContext::new("flattened-family-reader");
    let mut expected = Vec::new();
    for (name, per, offset) in [("looped", inner, 1u64), ("packed", 2, 100)] {
        let family = ring.family_artifact_input(
            production.clone(),
            name,
            outer * per,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        for member in 0..outer * per {
            let (i, j) = ((member / per) as u64, (member % per) as u64);
            let value = if name == "looped" { 16 * i + j + offset } else { 16 * i + 100 * (j + 1) };
            expected.push((format!("{name}_{member}"), value));
            consumer = consumer.output(format!("{name}_{member}"), family.at(member)).unwrap();
        }
    }
    let consumer = consumer
        .build()
        .unwrap()
        .validate_with_manifests(&ParamEnv::default(), &BTreeMap::from([(production, manifest)]))
        .unwrap();
    let outputs = execute_in_session(
        &consumer,
        &mut cpu_backend([cpu]),
        BTreeMap::new(),
        &mut store,
        rand::random(),
        ExecutionConfig::default(),
    )
    .unwrap()
    .outputs;
    for (name, value) in expected {
        let RuntimeValue::Matrix(matrix) = &outputs[&name] else { panic!("{name} is a matrix") };
        let coefficients = matrix.as_cpu_full().unwrap().entry(0, 0).coeffs_biguints();
        assert_eq!(coefficients[0], value.into(), "{name}");
        assert!(coefficients[1..].iter().all(Zero::is_zero), "{name}");
    }
}

/// A nested loop of a root lane returns two slices of one matrix and a member nothing reads; a
/// later loop of the lane reads both slices. Later waves copy the sliced matrix once, the unread
/// output keeps no members, and the result matches the CPU.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_nested_loop_slices_and_unread_outputs_replay() {
    let (cpu, gpu, ring) = parameters();
    let (outer, inner) = (2usize, 4usize);
    let sums = {
        let ring = ring.clone();
        parallel(outer, move |i| {
            let i = i.expression()?;
            let parts = parallel(inner, |j| {
                let j = j.expression()?;
                let stacked = Mat::concat(
                    ConcatAxis::Rows,
                    vec![
                        ring.polynomial([i.clone() * IntExpr::constant(10) + j.clone()]),
                        ring.polynomial([j.clone() + IntExpr::constant(3)]),
                    ],
                );
                let row = |row: usize| IndexRange {
                    start: IntExpr::constant(row),
                    end: IntExpr::constant(row + 1),
                };
                Ok((
                    stacked.clone().slice(Some(row(0)), None),
                    stacked.slice(Some(row(1)), None),
                    ring.polynomial([j + IntExpr::constant(7)]),
                ))
            })?;
            let (tops, bottoms) = (
                parts.field(|(top, _, _)| top.clone())?,
                parts.field(|(_, bottom, _)| bottom.clone())?,
            );
            Ok(tops.at(0usize) * bottoms.at(1usize) + tops.at(inner - 1) + bottoms.at(2usize))
        })
        .unwrap()
    };
    let mut context = DslContext::new("nested-slices");
    let names = (0..outer).map(|member| format!("sum_{member}")).collect::<Vec<_>>();
    for (member, name) in names.iter().enumerate() {
        context = context.output(name.clone(), sums.at(member)).unwrap();
    }
    let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    assert_matches_cpu(graph, BTreeMap::new(), &names, (cpu, gpu));
}

/// A loop reads a resident (computed) family at a runtime base plus its lane: the body selects
/// the members itself rather than importing them.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_resident_family_read_at_runtime_base() {
    let (cpu, gpu, ring) = parameters();
    let members = {
        let ring = ring.clone();
        parallel(
            8,
            move |member| Ok(ring.polynomial([member.expression()? + IntExpr::constant(5)])),
        )
        .unwrap()
    };
    let base = DslContext::new("base").input::<Int>("base", IntType).unwrap();
    let window = parallel(4, |lane| Ok(members.at(base.clone().add(lane)))).unwrap();
    let mut context = DslContext::new("resident-runtime-base");
    let names = (0..4).map(|lane| format!("window_{lane}")).collect::<Vec<_>>();
    for (lane, name) in names.iter().enumerate() {
        context = context.output(name.clone(), window.at(lane)).unwrap();
    }
    let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let inputs = BTreeMap::from([("base".to_owned(), RuntimeValue::Int(3.into()))]);
    assert_matches_cpu(graph, inputs, &names, (cpu, gpu));
}

/// Loop-dependent scalar powers (`3^i`) and bit positions (bit `i` of a runtime integer, at a
/// position whose interval bound reaches below zero) select their values on the device.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_loop_dependent_powers_and_bits() {
    let (cpu, gpu, ring) = parameters();
    let x = DslContext::new("x").input::<Int>("x", IntType).unwrap();
    let count = 5usize;
    let terms = {
        let ring = ring.clone();
        parallel(count, move |i| {
            let power = ring.constant(
                (1, 1),
                ConstantMatrix::PowerOfBase { base: 3.into(), exponent: i.expression()? },
            );
            // Bit `count - 1 - i`, and bit `(i - i) + i = i`, whose interval bound reaches below
            // zero through `i - i`.
            let high = x.clone().bit(Int::evaluate(IntExpr::Sub(
                Box::new(IntExpr::constant(count - 1)),
                Box::new(i.expression()?),
            )))?;
            let low = x.clone().bit(Int::evaluate(IntExpr::Add(
                Box::new(IntExpr::Sub(Box::new(i.expression()?), Box::new(i.expression()?))),
                Box::new(i.expression()?),
            )))?;
            let lift = |bit: mxx_dsl::Bool| {
                bit.to_int().lift_to_constant_polynomial(ring.matrix_type((1, 1)))
            };
            let low = lift(low);
            Ok(power + lift(high) + low.clone() + low)
        })
        .unwrap()
    };
    let mut context = DslContext::new("powers-and-bits");
    let names = (0..count).map(|i| format!("term_{i}")).collect::<Vec<_>>();
    for (i, name) in names.iter().enumerate() {
        context = context.output(name.clone(), terms.at(i)).unwrap();
    }
    let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let inputs = BTreeMap::from([("x".to_owned(), RuntimeValue::Int(0b10110.into()))]);
    assert_matches_cpu(graph, inputs, &names, (cpu, gpu));
}
