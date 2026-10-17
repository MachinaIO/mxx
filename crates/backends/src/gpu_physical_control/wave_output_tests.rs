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

/// Rematerializing a deterministic hash in a nested lane preserves its outer
/// index tag and observes new key inputs on a replay of the same GPU plan.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_parallel_hash_capture_matches_root() {
    let (_, gpu, ring) = parameters();
    let columns = crate::env::test_streamed_columns();
    let key = ring.bytes_input("hash-key", 32);
    let hash = |member: Int| {
        let mut tag = mxx_dsl::HashTag::from(b"nested-hash-capture".as_slice());
        tag.push_u64_le(member);
        ring.hash_matrix(key.clone(), tag, (1, columns))
    };
    let nested = parallel(2usize, |member| {
        let value = hash(member.clone());
        let other = hash(member.add(10usize));
        let lanes = parallel(3usize, |_| Ok(value.clone() + other.clone()))?;
        Ok((lanes.at(0usize), lanes.at(1usize), lanes.at(2usize)))
    })
    .unwrap();
    let mut graph = DslContext::new("nested-hash-capture");
    for member in 0..2 {
        graph = graph
            .output(
                format!("expected_{member}"),
                hash(Int::constant(member)) + hash(Int::constant(member + 10)),
            )
            .unwrap();
        let (a, b, c) = nested.at(member);
        for (lane, value) in [a, b, c].into_iter().enumerate() {
            graph = graph.output(format!("actual_{member}_{lane}"), value).unwrap();
        }
    }
    let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let inputs = || {
        BTreeMap::from([(
            "hash-key".to_owned(),
            RuntimeValue::Bytes(rand::random::<[u8; 32]>().to_vec().into()),
        )])
    };
    let first = inputs();
    let mut runtime = runtime(gpu);
    let mut plan = runtime.plan_with_fixed_geometry_for_test(graph, &first, 1, 1).unwrap();
    let mut store = MemoryArtifactStore::default();
    for values in [first, inputs()] {
        let result =
            runtime.execute_with_artifacts(&mut plan, values, &mut store, rand::random()).unwrap();
        for member in 0..2 {
            let expected = runtime
                .download_matrix_output(&result.output(&format!("expected_{member}")).unwrap())
                .unwrap();
            for lane in 0..3 {
                let actual = runtime
                    .download_matrix_output(
                        &result.output(&format!("actual_{member}_{lane}")).unwrap(),
                    )
                    .unwrap();
                assert_eq!(actual, expected, "member {member}, lane {lane}");
            }
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
    parallel_preimage_workspaces(1);
}

/// Sharing ordered tiles must never share native mutable workspace between
/// concurrently active parallel lanes, even when logical devices alias one GPU.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_parallel_preimage_workspace_independence() {
    parallel_preimage_workspaces(2);
}

fn parallel_preimage_workspaces(width: usize) {
    let (cpu, gpu, ring) = parameters();
    let digits = cpu.modulus_digits();
    let columns = crate::env::test_streamed_columns();
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
            let target = ring.uniform_residue((2, columns));
            let preimage = anchors
                .at(owner.clone())
                .sample_preimage(target.clone(), (2 * (digits + 2), columns));
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
        runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), width, 1).unwrap();
    // R and E must share the stacked leaf's storage in both sampled and dynamically
    // selected trapdoors; otherwise a family keeps a second copy of its largest payload.
    let mut shared = 0;
    for (id, owner) in plan.frame().owners.iter() {
        if !matches!(owner.wire_type(), mxx_ir_core::types::ConcreteWireType::Trapdoor {
            instantiation, ..
        } if instantiation.is_computational()) ||
            owner.physical().encodings.len() != 6
        {
            continue;
        }
        let placeholder =
            std::sync::Arc::new(owner.private_placeholder(runtime.backend()).unwrap());
        for owner in [std::sync::Arc::clone(owner), placeholder] {
            for part in owner.physical().parts.iter().filter(|part| part.leaf < 2) {
                assert!(
                    owner.physical().parts.iter().any(|stacked| {
                        let source = owner.storage(stacked.storage).unwrap();
                        let view = owner.storage(part.storage).unwrap();
                        stacked.leaf == 5 &&
                            view.address + part.view.byte_offset ==
                                source.address +
                                    stacked.view.byte_offset +
                                    u64::from(part.leaf) *
                                        part.view.extent[0] *
                                        stacked.view.byte_strides[0] &&
                            std::sync::Arc::ptr_eq(
                                &owner.storage(stacked.storage).unwrap().owner,
                                &owner.storage(part.storage).unwrap().owner,
                            )
                    }),
                    "R/E must alias the stacked allocation: {id:?} {:?}",
                    owner
                        .physical()
                        .parts
                        .iter()
                        .map(|p| (
                            p.leaf,
                            p.storage,
                            owner.storage(p.storage).unwrap().address,
                            p.view.byte_offset,
                            p.view.extent.clone(),
                            p.view.byte_strides.clone()
                        ))
                        .collect::<Vec<_>>()
                );
            }
        }
        shared += 1;
    }
    assert!(shared > 0, "the plan must exercise shared trapdoor leaves");
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
                (0..residual.size().0).all(|row| (0..columns).all(|column| residual
                    .entry(row, column)
                    .coeffs_biguints()
                    .iter()
                    .all(Zero::is_zero))),
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

/// Loop-dependent unit rows and columns place each iteration's block in its own block of a row
/// (and a column) inside a sequential loop and inside the lanes of a parallel loop.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_loop_dependent_unit_vectors_place_blocks() {
    let (cpu, gpu, ring) = parameters();
    let blocks = 5usize;
    let block = |index: Int| {
        let value = index.expression().unwrap() + IntExpr::constant(2);
        let entry = |scale: i64| ring.polynomial([value.clone() * IntExpr::constant(scale)]);
        Mat::concat(ConcatAxis::Columns, vec![entry(1), entry(2), entry(3)])
    };
    let unit = |shape: (usize, usize), index: &Int, row: bool| {
        let index = index.expression().unwrap();
        ring.constant(
            shape,
            if row {
                ConstantMatrix::UnitRow { index }
            } else {
                ConstantMatrix::UnitColumn { index }
            },
        )
    };
    let row = mxx_dsl::iterate(blocks, ring.zero((1, 3 * blocks)), |index, row| {
        Ok(row + unit((1, blocks), &index, true).tensor(block(index)))
    })
    .unwrap();
    let columns = parallel(2, |lane| {
        mxx_dsl::iterate(blocks, ring.zero((blocks, 3)), |index, column| {
            let shifted = index.clone().add(lane.clone()).rem(blocks);
            Ok(column + unit((blocks, 1), &shifted, false).tensor(block(index)))
        })
    })
    .unwrap();
    let graph = DslContext::new("unit-vectors")
        .output("row", row)
        .unwrap()
        .output("column_0", columns.at(0))
        .unwrap()
        .output("column_1", columns.at(1))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let names = ["row", "column_0", "column_1"].map(str::to_owned);
    assert_matches_cpu(graph, BTreeMap::new(), &names, (cpu, gpu));
}

/// A nested loop's family that its root lane both exports flattened and reads member by member
/// (in a loop of the lane whose lanes sum members in a sequential loop) is spilled: every member
/// reaches its flat index of the export, and the lane's sums read the members back.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_spilled_nested_family_is_exported_and_read_back() {
    let (cpu, gpu, ring) = parameters();
    let (outer, inner, halves) = (3usize, 4usize, 2usize);
    let instances = {
        let ring = ring.clone();
        parallel(outer, move |i| {
            let looped = parallel(inner, |j| {
                Ok(ring.polynomial([IntExpr::Add(
                    Box::new(IntExpr::Mul(Box::new(i.expression()?), Box::new(16.into()))),
                    Box::new(j.expression()? + IntExpr::constant(1)),
                )]))
            })?;
            let read = looped.clone();
            let ring = ring.clone();
            let halves = parallel(halves, move |half| {
                let read = read.clone();
                mxx_dsl::iterate(inner / 2, ring.zero((1, 1)), move |k, sum| {
                    Ok(sum + read.at(half.clone().mul(inner / 2).add(k)))
                })
            })?;
            Ok((looped, halves.at(0) + halves.at(1)))
        })
        .unwrap()
    };
    let looped = instances.field(|(looped, _)| looped.clone()).unwrap();
    let per = Int::evaluate(IntExpr::constant(inner));
    let flat = parallel(outer * inner, move |member| {
        Ok(looped.at(member.clone().div(per.clone())).at(member.rem(per.clone())))
    })
    .unwrap();
    let producer = DslContext::new("spilled-nested-family")
        .cached_output("looped", flat)
        .unwrap()
        .cached_output("sums", instances.field(|(_, sum)| sum.clone()).unwrap())
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let mut store = MemoryArtifactStore::default();
    let mut runtime = runtime(gpu.clone());
    let mut plan =
        runtime.plan_with_fixed_geometry_for_test(producer, &BTreeMap::new(), 1, 1).unwrap();
    // The members are written to the export and to a private scratch generation.
    let templates = &plan.physical_frame_for_test().export_templates;
    assert!(templates.iter().any(|t| t.name == "looped" && t.parent_stride == Some(inner)));
    assert!(templates.iter().any(|t| t.scratch && t.parent_stride == Some(inner)));
    let result = runtime
        .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
        .unwrap();
    let production = result.production_id.clone().expect("a producing execution");
    let manifest = store.load_finalized_manifest(&production).unwrap();
    let mut consumer = DslContext::new("spilled-family-reader");
    let mut expected = Vec::new();
    let looped = ring.family_artifact_input(
        production.clone(),
        "looped",
        outer * inner,
        (1, 1),
        ArtifactAvailability::Cached,
    );
    let sums = ring.family_artifact_input(
        production.clone(),
        "sums",
        outer,
        (1, 1),
        ArtifactAvailability::Cached,
    );
    for member in 0..outer * inner {
        let (i, j) = ((member / inner) as u64, (member % inner) as u64);
        expected.push((format!("looped_{member}"), 16 * i + j + 1));
        consumer = consumer.output(format!("looped_{member}"), looped.at(member)).unwrap();
    }
    for i in 0..outer {
        let i64 = i as u64;
        expected.push((format!("sum_{i}"), (0..inner as u64).map(|j| 16 * i64 + j + 1).sum()));
        consumer = consumer.output(format!("sum_{i}"), sums.at(i)).unwrap();
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

/// A failing conditional inside a wave must be checked before the next wave resets its status;
/// a subsequent successful replay of the same plan must not inherit the failed status.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_wave_control_failure_and_recovery() {
    let (_, gpu, ring) = parameters();
    let fault = DslContext::new("wave-control-input").input::<Int>("fault", IntType).unwrap();
    let rows = parallel(3, |lane| {
        let fault = fault.clone();
        let sum = mxx_dsl::iterate(2, lane, move |index, sum| {
            Ok(sum + Int::constant(1).div(Int::constant(3).sub(fault.clone()).sub(index)))
        })?;
        Ok(sum.lift_to_constant_polynomial(ring.matrix_type((1, 1))))
    })
    .unwrap();
    let graph = DslContext::new("wave-control-errors")
        .output("last", rows.at(2))
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let inputs = |fault: usize| BTreeMap::from([("fault".into(), RuntimeValue::Int(fault.into()))]);
    let mut runtime = runtime(gpu);
    runtime.options_mut().integer_input_ranges.insert("fault".into(), 0.into()..=2.into());
    let mut plan = runtime.plan_with_fixed_geometry_for_test(graph, &inputs(0), 1, 1).unwrap();
    for fault in [2, 0, 2] {
        let result = runtime.execute(&mut plan, inputs(fault));
        if fault == 0 {
            let output = result.unwrap();
            let value = runtime.download_matrix_output(&output.output("last").unwrap()).unwrap();
            assert_eq!(value.entry(0, 0).coeffs_biguints()[0], 2u8.into());
        } else {
            let error = result.err().expect("the second iteration must report division by zero");
            assert!(error.to_string().contains("division by zero"), "{error}");
        }
    }
}

/// Scalar results with a nested reduction use replayed waves rather than the elementwise
/// shortcut. Their later-wave owners and a dynamic gather must survive repeated execution.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_parallel_scalar_reduction_family() {
    let (cpu, gpu, ring) = parameters();
    let count = crate::env::test_streamed_members();
    let rows = parallel(count, |member| {
        mxx_dsl::iterate(3, member.clone(), |index, sum| {
            Ok(sum.add(member.clone().mul(index.add(1))))
        })
    })
    .unwrap();
    let initial = rows.clone();
    let rows = mxx_dsl::iterate(2, rows, |step, state| {
        parallel(count, |member| {
            mxx_dsl::iterate(3, state.at(member), |index, sum| Ok(sum.add(step.clone()).add(index)))
        })
    })
    .unwrap();
    let selected = DslContext::new("scalar-selection").input::<Int>("selected", IntType).unwrap();
    let output = rows.at(selected.clone()).lift_to_constant_polynomial(ring.matrix_type((1, 1)));
    let graph = DslContext::new("scalar-reduction-family")
        .output(
            "initial",
            initial.at(selected).lift_to_constant_polynomial(ring.matrix_type((1, 1))),
        )
        .unwrap()
        .output("out", output)
        .unwrap()
        .build()
        .unwrap()
        .validate(&ParamEnv::default())
        .unwrap();
    let index = rand::random::<u64>() as usize % count;
    let inputs = BTreeMap::from([("selected".into(), RuntimeValue::Int(index.into()))]);
    assert_matches_cpu(graph, inputs, &["initial".into(), "out".into()], (cpu, gpu));
}

/// Static views of nested wave results are copied into the outer lane's replay
/// destination. Both component outputs must survive subsequent outer waves.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_nested_parallel_component_results() {
    let (cpu, gpu, ring) = parameters();
    let count = crate::env::test_streamed_members();
    let rows = parallel(count, |member| {
        let parts = parallel(2usize, |component| {
            let scalar = member.clone().mul(2usize).add(component);
            let matrix = scalar.lift_to_constant_polynomial(ring.matrix_type((1, 1)));
            mxx_dsl::iterate(3usize, matrix.clone(), |_, sum| Ok(sum + matrix.clone()))
        })?;
        Ok((parts.at(0usize), parts.at(1usize)))
    })
    .unwrap();
    let mut graph = DslContext::new("nested-component-results");
    let mut names = Vec::new();
    for member in 0..count {
        let (left, right) = rows.at(member);
        for (component, row) in [left, right].into_iter().enumerate() {
            let name = format!("member_{member}_{component}");
            graph = graph.output(&name, row).unwrap();
            names.push(name);
        }
    }
    let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
    assert_matches_cpu(graph, BTreeMap::new(), &names, (cpu, gpu));
}

/// Nested component outputs remain live until the streamed carry has copied them.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_nested_component_streamed_carry() {
    let (cpu, gpu, ring) = parameters();
    let count = crate::env::test_streamed_members();
    let initial = parallel(count, |member| {
        let matrix = member.lift_to_constant_polynomial(ring.matrix_type((1, 1)));
        Ok((matrix.clone(), matrix))
    })
    .unwrap();
    let rows = mxx_dsl::iterate(2usize, initial, |_, previous| {
        parallel(count, |member| {
            let (left, right) = previous.at(member);
            let parts = parallel(2usize, |component| {
                let matrix = left.clone() +
                    right.clone() +
                    component.lift_to_constant_polynomial(ring.matrix_type((1, 1)));
                mxx_dsl::iterate(3usize, matrix.clone(), |_, sum| Ok(sum + matrix.clone()))
            })?;
            Ok((parts.at(0usize), parts.at(1usize)))
        })
    })
    .unwrap();
    let mut graph = DslContext::new("nested-component-streamed-carry");
    let mut names = Vec::new();
    for member in 0..count {
        let (left, right) = rows.at(member);
        for (component, row) in [left, right].into_iter().enumerate() {
            let name = format!("member_{member}_{component}");
            graph = graph.output(&name, row).unwrap();
            names.push(name);
        }
    }
    assert_matches_cpu(
        graph.build().unwrap().validate(&ParamEnv::default()).unwrap(),
        BTreeMap::new(),
        &names,
        (cpu, gpu),
    );
}

/// A runtime expression with a singleton range selects a view, without a matrix gather.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_singleton_index_uses_candidate() {
    let (cpu, gpu, ring) = parameters();
    let count = crate::env::test_streamed_members();
    let candidates = parallel(2usize, |i| {
        Ok(i.add(7usize).lift_to_constant_polynomial(ring.matrix_type((1, 1))))
    })
    .unwrap();
    let rows = parallel(count, |i| Ok(candidates.at(i.div(count)))).unwrap();
    let mut graph = DslContext::new("singleton-index-view");
    let mut names = Vec::new();
    for member in 0..count {
        let name = format!("member_{member}");
        graph = graph.output(&name, rows.at(member)).unwrap();
        names.push(name);
    }
    let graph = graph.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let plan = runtime(gpu.clone())
        .plan_with_fixed_geometry_for_test(graph.clone(), &BTreeMap::new(), 1, 1)
        .unwrap();
    assert!(plan.physical_frame_for_test().indexed_tables.is_empty());
    drop(plan);
    assert_matches_cpu(graph, BTreeMap::new(), &names, (cpu, gpu));
}

/// Capacity diagnostic for two resident anchors and a full-width compact preimage,
/// using the production planner at W=1, C=1. This is not a DIO2 round trip or a
/// complete DIO2 peak: its other live rows must be accounted separately.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_preimage_capacity() {
    preimage_capacity(0);
}

/// Include retained caller rows explicitly instead of assuming the sampler's
/// allocation ledger covers them. This is a synthetic capacity envelope, not a
/// complete DIO2 plan; its row count is overridable with MXX_TEST_STREAMED_MEMBERS.
#[test]
#[ignore = "requires a CUDA GPU"]
#[serial_test::serial(gpu_context)]
fn test_gpu_direct_preimage_capacity_retained_rows() {
    preimage_capacity(crate::env::test_streamed_members());
}

fn preimage_capacity(live_rows: usize) {
    let (n, primes, bits, base_bits) = crate::env::test_capacity_parameters();
    let cpu = DCRTPolyParams::new(n, primes, bits, base_bits, None, None);
    let gpu = GpuDCRTPolyParams::new(cpu.ring_dimension(), cpu.moduli().to_vec(), base_bits, None);
    let ring = Ring::from_crt_moduli(
        cpu.moduli().iter().copied().map(Into::into).collect(),
        cpu.ring_dimension(),
    );
    let digits = cpu.modulus_digits();
    let base = 1u32 << base_bits;
    let bound = crate::sampler::bounds::default_preimage_cutoff(
        cpu.ring_dimension(),
        2,
        digits,
        base,
        4.0,
        mxx_ir_core::types::TrapdoorInstantiation::Computational,
    )
    .unwrap();
    let anchors = parallel(2usize, |_| {
        Ok(ring.sample_trapdoor(2, 4, base, digits, BigInt::from(bound.clone())))
    })
    .unwrap();
    let rows = (live_rows != 0)
        .then(|| parallel(live_rows, |_| Ok(ring.uniform_residue((1, digits)))).unwrap());
    // Two refresh chunks of step zero: the production selector is instance / chunks.
    let preimages = parallel(2usize, |member| {
        let target = match &rows {
            Some(rows) => {
                let row = rows.at(member.clone().rem(live_rows));
                Mat::concat(mxx_ir_core::node::ConcatAxis::Rows, vec![row.clone(), row])
            }
            None => ring.uniform_residue((2, digits)),
        };
        Ok(anchors.at(member.div(2usize)).sample_preimage(target, (2 * (digits + 2), digits)))
    })
    .unwrap();
    let mut builder = DslContext::new("preimage-capacity")
        .cached_output("anchors", anchors.public_matrices())
        .unwrap()
        .cached_output("preimages", preimages)
        .unwrap();
    if let Some(rows) = rows {
        builder = builder.output("retained_rows", rows).unwrap();
    }
    let graph = builder.build().unwrap().validate(&ParamEnv::default()).unwrap();
    let plan =
        runtime(gpu).plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 1).unwrap();
    eprintln!(
        "capacity n={n} primes={primes} crt_bits={bits} base_bits={base_bits} digits={digits} bound_bits={} retained_rows={live_rows} W=1 C=1 planned_device_bytes={:?}",
        bound.bits(),
        plan.planned_device_bytes().unwrap()
    );
    // Logical views can overlap: this ledger identifies large owners, not an
    // additional peak estimate. Deferred graph scratch is reported separately
    // by planned_device_bytes after physical compilation.
    let mut allocations = BTreeMap::new();
    for (id, owner) in plan
        .frame()
        .owners
        .iter()
        .chain(plan.frame().waves.iter().flat_map(|wave| wave.owner_bindings.iter()))
    {
        for (_, bound) in owner.storages() {
            if crate::gpu_graph_memory::is_graph_managed(bound) {
                continue;
            }
            let entry = allocations.entry((bound.device, bound.address)).or_insert((0, Vec::new()));
            entry.0 = entry.0.max(bound.bytes);
            entry.1.push(*id);
        }
    }
    let mut allocations = allocations.into_iter().collect::<Vec<_>>();
    allocations.sort_by_key(|(_, (bytes, _))| std::cmp::Reverse(*bytes));
    for ((device, address), (bytes, values)) in allocations.into_iter().take(24) {
        let count = values.len();
        let examples = &values[..values.len().min(8)];
        eprintln!(
            "capacity allocation device={device} address={address} bytes={bytes} references={count} first_values={examples:?}"
        );
    }
}
