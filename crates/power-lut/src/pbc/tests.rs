use super::{
    schedule::{ValidatedSupport, deterministic_matching, schedule, schedule_from_owners},
    *,
};
use mxx_bgg::BggSamplerLayout;
use mxx_ir_core::artifact::{Manifest, ProductionId};

fn seed(byte: u8) -> PbcRootSeed {
    PbcRootSeed([byte; 32])
}

fn has_matching(layout: &PbcPublicLayout, support: &[usize]) -> bool {
    fn visit(layout: &PbcPublicLayout, support: &[usize], index: usize, used: &mut [bool]) -> bool {
        if index == support.len() {
            return true;
        }
        let coordinate = support[index];
        for &bucket in &layout.candidates[coordinate] {
            if !used[bucket] {
                used[bucket] = true;
                if visit(layout, support, index + 1, used) {
                    return true;
                }
                used[bucket] = false;
            }
        }
        false
    }
    visit(layout, support, 0, &mut vec![false; layout.parameters.bucket_count])
}

#[test]
fn parameter_profiles_and_boundaries() {
    let conservative = PbcParameters::conservative(10, 4);
    assert_eq!(conservative.hash_count, 3);
    assert_eq!(conservative.bucket_count, 6);
    assert_eq!(conservative.max_seed_attempts, 16);
    assert!(conservative.validate().is_ok());
    let paper = PbcParameters::paper_evaluation(10, 4);
    assert_eq!(paper.hash_count, 3);
    assert_eq!(paper.bucket_count, 7);
    assert_eq!(paper.max_seed_attempts, 128);
    assert!(paper.validate().is_ok());
    assert!(PbcParameters::custom(1, 1, 2, 2, 1, Some(2)).validate().is_ok());
    for parameters in [
        PbcParameters::custom(0, 1, 2, 2, 1, None),
        PbcParameters::custom(4, 0, 2, 2, 1, None),
        PbcParameters::custom(4, 5, 2, 5, 1, None),
        PbcParameters::custom(4, 2, 1, 2, 1, None),
        PbcParameters::custom(4, 3, 2, 2, 1, None),
        PbcParameters::custom(4, 1, 3, 2, 1, None),
        PbcParameters::custom(4, 2, 2, 1, 1, None),
        PbcParameters::custom(4, 2, 2, 2, 0, None),
        PbcParameters::custom(4, 2, 2, 2, 1, Some(1)),
    ] {
        assert!(parameters.validate().is_err());
    }
}

#[test]
fn named_profile_numeric_fields_are_bound_and_deserialization_is_validated() {
    for parameters in [PbcParameters::conservative(10, 4), PbcParameters::paper_evaluation(10, 4)] {
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(21), 0), 0).unwrap();
        for field in ["hash_count", "bucket_count", "max_seed_attempts"] {
            let mut forged = parameters.clone();
            match field {
                "hash_count" => forged.hash_count += 1,
                "bucket_count" => forged.bucket_count += 1,
                "max_seed_attempts" => forged.max_seed_attempts = 1,
                _ => unreachable!(),
            }
            assert!(forged.validate().is_err());
            assert!(PbcPublicLayout::build(&forged, derive_attempt_seed(seed(20), 0), 0).is_err());

            let mut tampered_parameters = serde_json::to_value(&parameters).unwrap();
            tampered_parameters[field] = match field {
                "max_seed_attempts" => serde_json::json!(1),
                "hash_count" => serde_json::json!(parameters.hash_count + 1),
                "bucket_count" => serde_json::json!(parameters.bucket_count + 1),
                _ => unreachable!(),
            };
            assert!(serde_json::from_value::<PbcParameters>(tampered_parameters).is_err());

            let mut tampered_layout = serde_json::to_value(&layout).unwrap();
            tampered_layout["parameters"][field] = match field {
                "max_seed_attempts" => serde_json::json!(1),
                "hash_count" => serde_json::json!(parameters.hash_count + 1),
                "bucket_count" => serde_json::json!(parameters.bucket_count + 1),
                _ => unreachable!(),
            };
            assert!(serde_json::from_value::<PbcPublicLayout>(tampered_layout).is_err());
        }
    }
}

#[test]
fn candidate_derivation_is_deterministic_distinct_and_in_range() {
    let a = derive_candidate_buckets(PbcLayoutSeed([7; 32]), 3, 11, 3).unwrap();
    let b = derive_candidate_buckets(PbcLayoutSeed([7; 32]), 3, 11, 3).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 3);
    assert!(a.iter().all(|&bucket| bucket < 11));
    assert!(a[0] != a[1] && a[0] != a[2] && a[1] != a[2]);
    assert_ne!(a, derive_candidate_buckets(PbcLayoutSeed([8; 32]), 3, 11, 3).unwrap());
    assert_ne!(a, derive_candidate_buckets(PbcLayoutSeed([7; 32]), 4, 11, 3).unwrap());
}

#[test]
fn layout_accounting_and_serialization_are_validated() {
    let parameters = PbcParameters::custom(9, 3, 3, 5, 4, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(1), 0), 0).unwrap();
    layout.validate().unwrap();
    assert_eq!(layout.cells.len(), parameters.bucket_count);
    assert!(layout.cells.iter().all(|row| row.len() == layout.bucket_width));
    assert_eq!(
        layout.cells.iter().flatten().filter(|cell| matches!(cell, PbcCell::Real { .. })).count(),
        parameters.universe_size * parameters.hash_count
    );
    assert_eq!(
        layout.cells.iter().flatten().filter(|cell| matches!(cell, PbcCell::Dummy)).count(),
        parameters.bucket_count
    );
    let encoded = serde_json::to_vec(&layout).unwrap();
    assert_eq!(serde_json::from_slice::<PbcPublicLayout>(&encoded).unwrap(), layout);
    let mut tampered = layout.clone();
    tampered.cells[0].swap(0, 1);
    assert!(tampered.validate().is_err());
    let mut wrong_id = layout.clone();
    wrong_id.layout_id = PbcLayoutId([0; 32]);
    assert!(wrong_id.validate().is_err());
    let mut wrong_version = layout.clone();
    wrong_version.semantic_version += 1;
    assert!(wrong_version.validate().is_err());
}

#[test]
fn malformed_private_schedule_cannot_select_padding() {
    let parameters = PbcParameters::custom(4, 1, 2, 3, 1, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(41), 0), 0).unwrap();
    let (padding_bucket, padding_slot) = layout
        .cells
        .iter()
        .enumerate()
        .find_map(|(bucket, row)| {
            row.iter().position(|cell| matches!(cell, PbcCell::Padding)).map(|slot| (bucket, slot))
        })
        .expect("fixture must contain rectangular padding");
    let mut selected_slots = layout
        .cells
        .iter()
        .map(|row| row.iter().position(|cell| matches!(cell, PbcCell::Dummy)).unwrap())
        .collect::<Vec<_>>();
    selected_slots[padding_bucket] = padding_slot;
    let malformed = PbcPrivateSchedule {
        layout_id: layout.layout_id,
        selected_slots,
        assigned_coordinates: vec![None; parameters.bucket_count],
        support_assignments: vec![(0, 0)],
    };
    assert!(matches!(malformed.validate(&layout), Err(PbcError::InvalidSchedule(_))));
}

#[test]
fn scheduler_matches_bruteforce_for_every_toy_support() {
    let parameters = PbcParameters::custom(6, 3, 2, 4, 1, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(2), 0), 0).unwrap();
    for a in 0..parameters.universe_size {
        for b in (a + 1)..parameters.universe_size {
            for c in (b + 1)..parameters.universe_size {
                let support = [a, b, c];
                let expected = has_matching(&layout, &support);
                let validated = ValidatedSupport::new(&parameters, &support).unwrap();
                let owners = deterministic_matching(&layout, &validated);
                assert_eq!(owners.is_ok(), expected, "support={support:?}");
                if let Ok(owners) = owners {
                    let scheduled = schedule_from_owners(&layout, &validated, owners).unwrap();
                    assert_eq!(canonical_decode(&layout, &scheduled).unwrap(), support.to_vec());
                }
                assert_eq!(schedule(&layout, &validated).is_ok(), expected, "support={support:?}");
            }
        }
    }
}

#[test]
fn support_validation_precedes_width_retry_for_generation_and_diagnostics() {
    let parameters = PbcParameters::custom(2, 2, 2, 2, 3, Some(2));
    for (support, expected) in [
        (&[][..], PbcError::SupportSize),
        (&[0, 0][..], PbcError::InvalidSupport),
        (&[0, 2][..], PbcError::InvalidSupport),
    ] {
        match generate_key_layout(&parameters, seed(44), support) {
            Err(error) => assert_eq!(error, expected.clone()),
            Ok(_) => panic!("invalid support was accepted"),
        }
        assert_eq!(measure_key_layout(&parameters, seed(44), support), Err(expected));
    }
}

#[test]
fn sorted_and_unsorted_supports_have_the_same_layout_and_schedule() {
    let parameters = PbcParameters::paper_evaluation(13, 3);
    let sorted = generate_key_layout(&parameters, seed(45), &[1, 7, 10]).unwrap();
    let unsorted = generate_key_layout(&parameters, seed(45), &[10, 1, 7]).unwrap();
    assert_eq!(sorted.public_layout, unsorted.public_layout);
    assert_eq!(
        sorted.private_schedule().selected_slots,
        unsorted.private_schedule().selected_slots
    );
    assert_eq!(
        sorted.private_schedule().assigned_coordinates,
        unsorted.private_schedule().assigned_coordinates
    );
    assert_eq!(
        canonical_decode(&sorted.public_layout, sorted.private_schedule()).unwrap(),
        canonical_decode(&unsorted.public_layout, unsorted.private_schedule()).unwrap()
    );

    let sorted_sample = measure_key_layout(&parameters, seed(45), &[1, 7, 10]).unwrap();
    let unsorted_sample = measure_key_layout(&parameters, seed(45), &[10, 1, 7]).unwrap();
    assert_eq!(sorted_sample.accepted_attempt, unsorted_sample.accepted_attempt);
    assert_eq!(sorted_sample.bucket_width, unsorted_sample.bucket_width);
    assert_eq!(sorted_sample.bucket_width_failures, unsorted_sample.bucket_width_failures);
    assert_eq!(
        sorted_sample.no_perfect_schedule_failures,
        unsorted_sample.no_perfect_schedule_failures
    );
}

#[test]
fn diagnostic_outcome_matches_key_layout_generation() {
    for (parameters, support) in [
        (PbcParameters::paper_evaluation(13, 3), vec![10, 1, 7]),
        (PbcParameters::custom(2, 1, 2, 2, 3, Some(2)), vec![0]),
    ] {
        let sample = measure_key_layout(&parameters, seed(46), &support).unwrap();
        match generate_key_layout(&parameters, seed(46), &support) {
            Ok(generated) => {
                assert_eq!(sample.accepted_attempt, Some(generated.public_layout.accepted_attempt));
                assert_eq!(sample.bucket_width, Some(generated.public_layout.bucket_width));
                assert!(sample.accepted());
            }
            Err(PbcError::SeedAttemptsExhausted(diagnostics)) => {
                assert_eq!(sample.accepted_attempt, None);
                assert_eq!(sample.attempts, diagnostics.attempts);
                assert_eq!(sample.bucket_width_failures, diagnostics.bucket_width_failures);
                assert_eq!(
                    sample.no_perfect_schedule_failures,
                    diagnostics.no_perfect_schedule_failures
                );
            }
            Err(error) => panic!("unexpected generation error: {error:?}"),
        }
    }
}

#[test]
fn generated_layout_retries_deterministically_and_oracle_preserves_inner_product() {
    let parameters = PbcParameters::paper_evaluation(13, 3);
    let generated = generate_key_layout(&parameters, seed(3), &[10, 1, 7]).unwrap();
    let generated_again = generate_key_layout(&parameters, seed(3), &[10, 1, 7]).unwrap();
    assert_eq!(generated.public_layout, generated_again.public_layout);
    assert_eq!(
        format!("{:?}", generated.private_schedule()),
        format!("{:?}", generated_again.private_schedule())
    );
    assert_eq!(
        canonical_decode(&generated.public_layout, generated.private_schedule()).unwrap(),
        vec![1, 7, 10]
    );
    let vector = (0..13).map(|x| (x * 17 + 5) as u64).collect::<Vec<_>>();
    let result = clear_pbc_inner_product(
        &generated.public_layout,
        generated.private_schedule(),
        &vector,
        97,
    )
    .unwrap();
    assert_eq!(result, (vector[1] + vector[7] + vector[10]) % 97);
}

#[test]
fn oracle_matches_direct_sparse_sum_across_layouts_and_dummy_buckets() {
    let parameters = PbcParameters::custom(7, 2, 3, 4, 8, None);
    let vector = [5_u64, 17, 29, 41, 53, 65, 77];
    for seed_byte in 0_u8..8 {
        for first in 0..parameters.universe_size {
            for second in (first + 1)..parameters.universe_size {
                let generated =
                    generate_key_layout(&parameters, seed(seed_byte), &[first, second]).unwrap();
                assert!(
                    generated.private_schedule().assigned_coordinates.iter().any(Option::is_none),
                    "k > h must route at least one bucket to its dummy"
                );
                assert_eq!(
                    clear_pbc_inner_product(
                        &generated.public_layout,
                        generated.private_schedule(),
                        &vector,
                        19,
                    )
                    .unwrap(),
                    (vector[first] + vector[second]) % 19,
                );
            }
        }
    }
}

#[test]
fn dense_binary_support_rejects_nonbinary_and_wrong_length() {
    assert_eq!(dense_binary_support(4, &[0, 1, 0, 1]).unwrap(), vec![1, 3]);
    assert!(dense_binary_support(4, &[0, 1]).is_err());
    assert!(dense_binary_support(4, &[0, 2, 0, 1]).is_err());
    let parameters = PbcParameters::custom(4, 2, 2, 2, 1, None);
    assert_eq!(support_from_dense(&parameters, &[0, 1, 0, 1]).unwrap(), vec![1, 3]);
    assert!(support_from_dense(&parameters, &[0, 1, 0, 0]).is_err());
}

#[test]
fn seed_exhaustion_reports_only_public_aggregate_diagnostics() {
    let parameters = PbcParameters::custom(2, 1, 2, 2, 3, Some(2));
    let first = generate_key_layout(&parameters, seed(9), &[0]).unwrap_err();
    let second = generate_key_layout(&parameters, seed(9), &[0]).unwrap_err();
    assert_eq!(first, second);
    match first {
        PbcError::SeedAttemptsExhausted(diagnostics) => {
            assert_eq!(diagnostics.attempts, 3);
            assert_eq!(diagnostics.bucket_width_failures, 3);
            assert_eq!(diagnostics.no_perfect_schedule_failures, 0);
            assert_eq!(diagnostics.last_public_cause, Some(PbcRetryCause::BucketWidthExceeded));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn encoded_public_vector_routes_labels_without_reusing_pbc_hash_domain() {
    let parameters = PbcParameters::custom(7, 2, 2, 3, 1, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(10), 0), 0).unwrap();
    let vector =
        PbcEncodedPublicVector::route(&layout, &[0u64, 11, 22, 33, 44, 55, 66], 17).unwrap();
    vector.validate(&layout).unwrap();
    assert_eq!(vector.layout_id, layout.layout_id);
    assert_eq!(vector.modulus, 17);
    for (bucket, row) in layout.cells.iter().enumerate() {
        for (slot, cell) in row.iter().enumerate() {
            match cell {
                PbcCell::Real { coordinate, .. } => {
                    assert_eq!(
                        vector.values[bucket][slot],
                        ((coordinate * 11) as u64 % 17) as usize
                    );
                }
                PbcCell::Dummy | PbcCell::Padding => assert_eq!(vector.values[bucket][slot], 0),
            }
        }
    }
    let a = PbcEncodedPublicVector::from_label(&layout, b"label-a", 17).unwrap();
    let b = PbcEncodedPublicVector::from_label(&layout, b"label-b", 17).unwrap();
    assert_ne!(a.values, b.values);
    assert_eq!(derive_lwr_vector(layout.layout_id, b"label-a", 7, 17).unwrap().len(), 7);
    assert_ne!(
        derive_lwr_vector(layout.layout_id, b"label-a", 7, 17).unwrap(),
        derive_lwr_vector(PbcLayoutId([9; 32]), b"label-a", 7, 17).unwrap()
    );
    assert!(a.values.iter().flatten().all(|&value| value < 17));
}

#[test]
fn selector_names_are_layout_and_key_namespace_bound_but_schedule_independent() {
    use crate::rhs::PowerRhsPackageArtifactNames;
    let parameters = PbcParameters::custom(7, 2, 2, 3, 1, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(11), 0), 0).unwrap();
    let package_names = |key_instance_id| {
        layout
            .cells
            .iter()
            .enumerate()
            .flat_map(|(bucket, row)| {
                row.iter().enumerate().filter_map(move |(slot, cell)| {
                    (!matches!(cell, PbcCell::Padding)).then_some(PbcSelectorPackageArtifactNames {
                        bucket,
                        slot,
                        package: PowerRhsPackageArtifactNames {
                            gsw_ciphertext: canonical_component_name(
                                layout.layout_id,
                                key_instance_id,
                                bucket,
                                slot,
                            ),
                        },
                    })
                })
            })
            .collect::<Vec<_>>()
    };
    let names_a =
        PbcSelectorArtifactNames::canonicalize(&layout, [1; 32], package_names([1; 32])).unwrap();
    let names_b =
        PbcSelectorArtifactNames::canonicalize(&layout, [2; 32], package_names([2; 32])).unwrap();
    assert_eq!(
        names_a.selector_packages.len(),
        parameters.universe_size * parameters.hash_count + parameters.bucket_count
    );
    assert_ne!(names_a, names_b);
    assert!(
        names_a
            .selector_packages
            .iter()
            .all(|entry| !entry.package.gsw_ciphertext.contains("selected"))
    );
    assert!(
        names_a
            .selector_packages
            .iter()
            .all(|entry| !entry.package.gsw_ciphertext.contains("support"))
    );
}

#[test]
fn selector_artifact_import_rejects_production_mismatch() {
    use mxx_ir_core::artifact::{Manifest, ProductionId, SpecHash};

    let parameters = PbcParameters::custom(7, 2, 2, 3, 1, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(12), 0), 0).unwrap();
    let names = PbcSelectorArtifactNames { selector_packages: Vec::new() };
    let expected = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
    let wrong_manifest = Manifest {
        ir_version: 1,
        production_id: ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] },
        artifacts: Default::default(),
    };
    let sampler_layout = BggSamplerLayout {
        modulus: 257.into(),
        ring_dimension: 4.into(),
        secret_dimension: 2,
        digit_count: 1,
        gadget_base: 2.into(),
    };
    assert!(
        PbcSelectorArtifacts::import(
            expected,
            &layout,
            [4; 32],
            &wrong_manifest,
            names,
            &sampler_layout,
            [4; 32],
            &sampler_layout,
            [4; 32],
        )
        .is_err()
    );
}

#[test]
fn trusted_selector_bits_validate_count_order_and_binary_values() {
    use mxx_bgg::BggSamplerLayout;

    let parameters = PbcParameters::custom(5, 1, 2, 3, 1, None);
    let generated = generate_key_layout(&parameters, seed(15), &[0]).unwrap();
    let layout = &generated.public_layout;
    let sampler_layout = BggSamplerLayout {
        modulus: 257.into(),
        ring_dimension: 4.into(),
        secret_dimension: 2,
        digit_count: 1,
        gadget_base: 2.into(),
    };
    let ring = sampler_layout.ring();
    let active = PbcActiveCellIndex::build(layout).unwrap();
    let expected_active = layout
        .cells
        .iter()
        .enumerate()
        .flat_map(|(bucket, row)| {
            row.iter().enumerate().filter_map(move |(slot, cell)| {
                (!matches!(cell, PbcCell::Padding)).then_some((bucket, slot))
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        active.iter().map(|(bucket, slot, _)| (bucket, slot)).collect::<Vec<_>>(),
        expected_active
    );
    assert!(
        active
            .iter()
            .all(|(bucket, slot, _)| { !matches!(layout.cells[bucket][slot], PbcCell::Padding) })
    );
    assert!(layout.cells.iter().enumerate().all(|(bucket, row)| {
        let dummy = row.iter().position(|cell| matches!(cell, PbcCell::Dummy)).unwrap();
        active.bucket_iter(bucket).unwrap().last().unwrap().1 == dummy
    }));
    let mut bits = vec![0_u8; active.len()];
    for (bucket, slot, flat) in active.iter() {
        if generated.private_schedule().selected_slot(bucket) == slot {
            bits[flat] = 1;
        }
    }
    assert!(
        PbcTrustedSelectorBits::from_host_bits(
            layout,
            generated.private_schedule(),
            &ring,
            [6; 32],
            &bits[..bits.len() - 1],
        )
        .is_err()
    );
    let mut nonbinary = bits.clone();
    nonbinary[0] = 2;
    assert!(
        PbcTrustedSelectorBits::from_host_bits(
            layout,
            generated.private_schedule(),
            &ring,
            [6; 32],
            &nonbinary,
        )
        .is_err()
    );
    let mut wrong_order = bits.clone();
    let first = active
        .iter()
        .find(|(bucket, slot, _)| generated.private_schedule().selected_slot(*bucket) == *slot)
        .unwrap()
        .2;
    let first_bucket = active.iter().find(|(_, _, flat)| *flat == first).unwrap().0;
    let second = active
        .iter()
        .find(|(bucket, slot, _)| {
            *bucket != first_bucket && generated.private_schedule().selected_slot(*bucket) != *slot
        })
        .unwrap()
        .2;
    wrong_order.swap(first, second);
    assert!(
        PbcTrustedSelectorBits::from_host_bits(
            layout,
            generated.private_schedule(),
            &ring,
            [6; 32],
            &wrong_order,
        )
        .is_err()
    );
    let trusted = PbcTrustedSelectorBits::from_schedule(&generated, &ring, [6; 32]).unwrap();
    assert_eq!(trusted.runtime_values().len(), active.len());
    assert_eq!(trusted.family().count(), &mxx_ir_core::IntExpr::constant(active.len()));
    assert!(!trusted.input_name().contains("support"));
}

#[test]
fn selector_sampling_uses_one_structural_loop_and_runtime_bits() {
    use crate::encoding::PowerLutEncodingSampler;
    use mxx_bgg::BggSamplerLayout;
    use mxx_dsl::DslContext;
    use mxx_ir_core::node::NodeKind;

    let parameters = PbcParameters::custom(5, 1, 2, 3, 1, None);
    let first = generate_key_layout(&parameters, seed(16), &[0]).unwrap();
    let second = generate_key_layout(&parameters, seed(16), &[1]).unwrap();
    assert_eq!(first.public_layout, second.public_layout);
    let sampler_layout = BggSamplerLayout {
        modulus: 257.into(),
        ring_dimension: 4.into(),
        secret_dimension: 2,
        digit_count: 1,
        gadget_base: 2.into(),
    };
    let ring = sampler_layout.ring();
    let sampler = PowerLutEncodingSampler {
        layout: sampler_layout.clone(),
        gaussian_sigma: None,
        gaussian_max_coefficient_bound: None,
    };
    let bits = PbcTrustedSelectorBits::from_schedule(&first, &ring, [7; 32]).unwrap();
    let source = sampler.sample_secret().unwrap();
    let target = sampler.sample_secret().unwrap();
    let hash_key = ring.bytes_input("selector-hash-key", 32);
    let families = build_structural_selector_families(
        &sampler,
        bits.family().clone(),
        source,
        target,
        hash_key,
        &first.public_layout,
        [7; 32],
    )
    .unwrap();
    let names = PbcSelectorArtifactNames::canonicalize_schema(
        &first.public_layout,
        [7; 32],
        sampler_layout.secret_dimension,
        sampler_layout.public_key_columns(),
    )
    .unwrap();
    let artifacts = PbcSelectorArtifacts::from_structural(
        &first.public_layout,
        [7; 32],
        names,
        &sampler_layout,
        [7; 32],
        &sampler_layout,
        [7; 32],
    )
    .unwrap();
    let context = artifacts
        .add_structural_family_outputs(
            DslContext::new("trusted-selector-structural-loop"),
            &first.public_layout,
            families,
        )
        .unwrap();
    let built = context.build().unwrap();
    let loops = built
        .graph
        .root_scope()
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind(), NodeKind::ParallelGrid(_)))
        .count();
    assert_eq!(loops, 1);
    let bit_name = bits.input_name().to_owned();
    let bit_inputs = built.graph.root_scope().nodes().iter().filter(|node| {
        matches!(node.kind(), NodeKind::Input { name, artifact: None, .. } if name == &bit_name)
    });
    assert_eq!(bit_inputs.count(), 1);

    // A different private schedule still produces the same structural graph
    // schema.  Concrete fixed-C family contents are runtime artifacts and may
    // differ with the sampled RHS inputs; this assertion concerns only the
    // graph shape and public namespace, not matrix-value invariance.
    let second_bits = PbcTrustedSelectorBits::from_schedule(&second, &ring, [7; 32]).unwrap();
    let second_families = build_structural_selector_families(
        &sampler,
        second_bits.family().clone(),
        sampler.sample_secret().unwrap(),
        sampler.sample_secret().unwrap(),
        ring.bytes_input("selector-hash-key", 32),
        &second.public_layout,
        [7; 32],
    )
    .unwrap();
    let second_names = PbcSelectorArtifactNames::canonicalize_schema(
        &second.public_layout,
        [7; 32],
        sampler_layout.secret_dimension,
        sampler_layout.public_key_columns(),
    )
    .unwrap();
    let second_artifacts = PbcSelectorArtifacts::from_structural(
        &second.public_layout,
        [7; 32],
        second_names,
        &sampler_layout,
        [7; 32],
        &sampler_layout,
        [7; 32],
    )
    .unwrap();
    let second_context = second_artifacts
        .add_structural_family_outputs(
            DslContext::new("trusted-selector-structural-loop"),
            &second.public_layout,
            second_families,
        )
        .unwrap();
    let second_built = second_context.build().unwrap();
    let first_hash = mxx_ir_core::encoding::spec_hash(&built.graph, &Default::default()).unwrap();
    let second_hash =
        mxx_ir_core::encoding::spec_hash(&second_built.graph, &Default::default()).unwrap();
    assert_eq!(first_hash, second_hash);
}
#[test]
fn fixed_rhs_selector_schema_exposes_only_gsw_component() {
    let parameters = PbcParameters::custom(7, 2, 2, 3, 1, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(14), 0), 0).unwrap();
    let names = PbcSelectorArtifactNames::canonicalize_schema(&layout, [8; 32], 1, 1).unwrap();
    assert_eq!(
        names.selector_packages.len(),
        parameters.universe_size * parameters.hash_count + parameters.bucket_count
    );
    assert!(
        names
            .selector_packages
            .iter()
            .all(|package| !package.package.gsw_ciphertext.contains("selected"))
    );
    assert!(
        names
            .selector_packages
            .iter()
            .all(|package| !package.package.gsw_ciphertext.contains("support"))
    );
    let sampler_layout = BggSamplerLayout {
        modulus: 257.into(),
        ring_dimension: 4.into(),
        secret_dimension: 2,
        digit_count: 1,
        gadget_base: 2.into(),
    };
    let artifacts = PbcSelectorArtifacts::from_structural(
        &layout,
        [8; 32],
        names.clone(),
        &sampler_layout,
        [8; 32],
        &sampler_layout,
        [8; 32],
    )
    .unwrap();
    let family_name = selector_family_artifact_name(&artifacts, "gsw");
    assert!(!family_name.is_empty());
}

fn selector_family_manifest_fixture() -> (
    PbcSelectorArtifacts,
    PbcPublicLayout,
    PbcSelectorArtifactNames,
    BggSamplerLayout,
    ProductionId,
    Manifest,
    [u8; 32],
    [u8; 32],
    [u8; 32],
) {
    use mxx_ir_core::{
        artifact::{ArtifactConfidentiality, ArtifactType, ManifestArtifact, SpecHash},
        types::ConcreteMatrixType,
    };
    use std::collections::BTreeMap;

    let parameters = PbcParameters::custom(5, 1, 2, 2, 1, None);
    let layout = PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(17), 0), 0).unwrap();
    let sampler = BggSamplerLayout {
        modulus: 97.into(),
        ring_dimension: 4.into(),
        secret_dimension: 2,
        digit_count: 1,
        gadget_base: 4.into(),
    };
    let key_instance_id = [4; 32];
    let source_identity = [5; 32];
    let target_identity = [6; 32];
    let names = PbcSelectorArtifactNames::canonicalize_schema(
        &layout,
        key_instance_id,
        sampler.secret_dimension,
        sampler.public_key_columns(),
    )
    .unwrap();
    let artifacts = PbcSelectorArtifacts::from_structural(
        &layout,
        key_instance_id,
        names.clone(),
        &sampler,
        source_identity,
        &sampler,
        target_identity,
    )
    .unwrap();
    let family_name = selector_family_artifact_name(&artifacts, "gsw");
    let active_count = names.selector_packages.len();
    let mut manifest = Manifest {
        ir_version: mxx_ir_core::encoding::IR_VERSION,
        production_id: ProductionId { spec_hash: SpecHash([7; 32]), execution_nonce: [8; 32] },
        artifacts: BTreeMap::from([(
            family_name,
            ManifestArtifact {
                artifact_type: ArtifactType::Matrix(ConcreteMatrixType {
                    modulus: 97.into(),
                    ring_dimension: 4,
                    rows: sampler.secret_dimension,
                    columns: sampler.public_key_columns(),
                }),
                family_shape: Some(vec![active_count]),
                confidentiality: ArtifactConfidentiality::Public,
                content_hash: Some([9; 32]),
                layout: None,
            },
        )]),
    };
    artifacts.finalize_export_manifest(&mut manifest).unwrap();
    (
        artifacts,
        layout,
        names,
        sampler,
        manifest.production_id.clone(),
        manifest,
        key_instance_id,
        source_identity,
        target_identity,
    )
}

#[test]
fn selector_family_finalization_preserves_runtime_descriptor() {
    let (artifacts, _, _, _, _, mut manifest, _, _, _) = selector_family_manifest_fixture();
    let before = manifest.clone();
    artifacts.finalize_export_manifest(&mut manifest).unwrap();
    assert_eq!(manifest, before);
    assert!(manifest.artifacts.values().all(|descriptor| descriptor.layout.is_none()));
}

#[test]
fn selector_family_import_round_trip_validates_rhs_layouts_and_identities() {
    let (_, layout, names, sampler, production_id, manifest, key_id, source_id, target_id) =
        selector_family_manifest_fixture();
    let imported = PbcSelectorArtifacts::import(
        production_id,
        &layout,
        key_id,
        &manifest,
        names,
        &sampler,
        source_id,
        &sampler,
        target_id,
    )
    .unwrap();
    assert_eq!(imported.layout_id(), layout.layout_id);
    assert_eq!(imported.key_instance_id(), key_id);
    assert_eq!(imported.package_count(), layout.parameters.universe_size * 2 + 2);
}

#[test]
fn selector_family_import_rejects_wrong_rhs_bindings_and_family_schema() {
    use mxx_ir_core::{artifact::ArtifactType, types::ConcreteMatrixType};

    let (_, layout, names, sampler, production_id, manifest, key_id, source_id, target_id) =
        selector_family_manifest_fixture();
    let import = |manifest: &Manifest,
                  names: PbcSelectorArtifactNames,
                  layout: &PbcPublicLayout,
                  key_id: [u8; 32],
                  source: &BggSamplerLayout,
                  source_id: [u8; 32],
                  target: &BggSamplerLayout,
                  target_id: [u8; 32]| {
        PbcSelectorArtifacts::import(
            production_id.clone(),
            layout,
            key_id,
            manifest,
            names,
            source,
            source_id,
            target,
            target_id,
        )
    };

    let mut wrong_source = sampler.clone();
    wrong_source.gadget_base = 8.into();
    assert!(
        import(
            &manifest,
            names.clone(),
            &layout,
            key_id,
            &wrong_source,
            source_id,
            &sampler,
            target_id
        )
        .is_err()
    );
    assert!(
        import(&manifest, names.clone(), &layout, key_id, &sampler, [10; 32], &sampler, target_id)
            .is_err()
    );
    let mut wrong_target = sampler.clone();
    wrong_target.digit_count = 2;
    assert!(
        import(
            &manifest,
            names.clone(),
            &layout,
            key_id,
            &sampler,
            source_id,
            &wrong_target,
            target_id
        )
        .is_err()
    );
    assert!(
        import(
            &manifest,
            names.clone(),
            &layout,
            [11; 32],
            &sampler,
            source_id,
            &sampler,
            target_id
        )
        .is_err()
    );

    let mut wrong_count = manifest.clone();
    let family_name = selector_family_artifact_name(
        &PbcSelectorArtifacts::from_structural(
            &layout,
            key_id,
            names.clone(),
            &sampler,
            source_id,
            &sampler,
            target_id,
        )
        .unwrap(),
        "gsw",
    );
    wrong_count.artifacts.get_mut(&family_name).unwrap().family_shape = Some(vec![1]);
    assert!(
        import(
            &wrong_count,
            names.clone(),
            &layout,
            key_id,
            &sampler,
            source_id,
            &sampler,
            target_id
        )
        .is_err()
    );

    let mut wrong_type = manifest.clone();
    wrong_type.artifacts.get_mut(&family_name).unwrap().artifact_type =
        ArtifactType::Matrix(ConcreteMatrixType {
            modulus: 97.into(),
            ring_dimension: 4,
            rows: 2,
            columns: 1,
        });
    assert!(
        import(
            &wrong_type,
            names.clone(),
            &layout,
            key_id,
            &sampler,
            source_id,
            &sampler,
            target_id
        )
        .is_err()
    );

    let mut wrong_metadata = manifest.clone();
    let descriptor = wrong_metadata.artifacts.get_mut(&family_name).unwrap();
    // The runtime owns the generic descriptor layout and persists it as
    // `None`.  A PBC-specific layout payload is therefore malformed rather
    // than a second source of metadata that import would have to reconcile.
    descriptor.layout = Some("unexpected-pbc-metadata".to_owned());
    assert!(
        import(
            &wrong_metadata,
            names.clone(),
            &layout,
            key_id,
            &sampler,
            source_id,
            &sampler,
            target_id
        )
        .is_err()
    );

    let mut wrong_order_names = names;
    wrong_order_names.selector_packages.swap(0, 1);
    assert!(
        import(
            &manifest,
            wrong_order_names,
            &layout,
            key_id,
            &sampler,
            source_id,
            &sampler,
            target_id
        )
        .is_err()
    );
}
