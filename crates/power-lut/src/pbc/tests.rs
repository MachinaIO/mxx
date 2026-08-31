use super::{
    matching::deterministic_matching,
    schedule::{ValidatedSupport, schedule, schedule_from_owners},
    *,
};
use serial_test::serial;

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
                                b"gsw",
                                0,
                                0,
                            ),
                            companions: Vec::new(),
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
    assert!(
        PbcSelectorArtifacts::import(expected, &layout, [4; 32], &wrong_manifest, names,).is_err()
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
    let artifacts =
        PbcSelectorArtifacts::from_structural(&first.public_layout, [7; 32], names).unwrap();
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
        .filter(|node| matches!(node.kind(), NodeKind::ParallelLoop(_)))
        .count();
    assert_eq!(loops, 1);
    let bit_name = bits.input_name().to_owned();
    let bit_inputs = built.graph.root_scope().nodes().iter().filter(|node| {
        matches!(node.kind(), NodeKind::Input { name, artifact: None, .. } if name == &bit_name)
    });
    assert_eq!(bit_inputs.count(), 1);

    // A different private schedule changes only runtime values.  Rebuilding
    // the same graph namespace therefore yields the same graph specification;
    // in particular public companion matrix expressions do not depend on the
    // hidden schedule choice.
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
    let second_artifacts =
        PbcSelectorArtifacts::from_structural(&second.public_layout, [7; 32], second_names)
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
#[serial(dcrt_runtime)]
fn selector_import_accepts_a_repository_valid_manifest() {
    use crate::{
        BggEncodingWire, BggPublicKeyWire,
        encoding::{BggEncodingArtifactNames, power_encoding_artifact_layout},
        pbc::compiler::PbcSelectorFamilyInputs,
        rhs::{
            ManifestSecretMetadata, PowerRhsCompanionArtifactName, PowerRhsPackageArtifactNames,
            power_rhs_artifact_layout,
        },
    };
    use mxx_bgg::BggSamplerLayout;
    use mxx_dsl::{DslContext, Int};
    use mxx_ir_core::{
        artifact::{
            ArtifactConfidentiality, ArtifactType, Manifest, ManifestArtifact, ProductionId,
            SpecHash,
        },
        types::ConcreteMatrixType,
    };
    use mxx_primitives::{
        matrix::PolyMatrix,
        poly::{Poly, PolyParams},
    };
    use mxx_runtime::{
        RuntimeValue,
        artifact::{ArtifactKey, ArtifactPayload, ArtifactStore},
        backend::{Backend, poly::cpu_backend},
        execute,
        transcript::SamplingMode,
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    let parameters = PbcParameters::custom(1, 1, 2, 2, 1, None);
    let generated = generate_key_layout(&parameters, seed(14), &[0]).unwrap();
    let layout = &generated.public_layout;
    let key_instance_id = [8; 32];
    let production_id = ProductionId { spec_hash: SpecHash([9; 32]), execution_nonce: [10; 32] };
    let parameters = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(4, 1, 17, 17);
    let modulus = BigInt::from(parameters.modulus().as_ref().clone());
    let sampler = BggSamplerLayout {
        modulus: modulus.clone().into(),
        ring_dimension: 4.into(),
        secret_dimension: 1,
        digit_count: 1,
        gadget_base: 2.into(),
    };
    let identity = [11; 32];
    let secret_metadata = ManifestSecretMetadata {
        modulus: sampler.modulus.clone(),
        ring_dimension: sampler.ring_dimension.clone(),
        secret_dimension: sampler.secret_dimension,
        digit_count: sampler.digit_count,
        gadget_base: sampler.gadget_base.clone(),
        identity,
    };
    let ring = sampler.ring();
    let matrix = ArtifactType::Matrix(ConcreteMatrixType {
        modulus: modulus.clone(),
        ring_dimension: 4,
        rows: 1,
        columns: 1,
    });
    let descriptor =
        |artifact_type: ArtifactType, confidentiality: ArtifactConfidentiality, layout: String| {
            ManifestArtifact {
                artifact_type,
                family_count: None,
                confidentiality,
                content_hash: None,
                layout: Some(layout),
            }
        };
    let mut package_names = Vec::new();
    let mut artifacts = BTreeMap::new();
    for (bucket, row) in layout.cells.iter().enumerate() {
        for (slot, cell) in row.iter().enumerate() {
            if matches!(cell, PbcCell::Padding) {
                continue;
            }
            let coordinate = match cell {
                PbcCell::Real { coordinate, .. } => Some(*coordinate),
                PbcCell::Dummy => None,
                PbcCell::Padding => unreachable!(),
            };
            let role = serde_json::json!({
                "PbcSelectorBit": {
                    "layout": layout.layout_id.0,
                    "bucket": bucket,
                    "slot": slot,
                    "coordinate": coordinate,
                }
            });
            let gsw_name = canonical_component_name(
                layout.layout_id,
                key_instance_id,
                bucket,
                slot,
                b"gsw",
                0,
                0,
            );
            let vector_name = canonical_component_name(
                layout.layout_id,
                key_instance_id,
                bucket,
                slot,
                b"vector",
                0,
                0,
            );
            let public_name = canonical_component_name(
                layout.layout_id,
                key_instance_id,
                bucket,
                slot,
                b"public",
                0,
                0,
            );
            artifacts.insert(
                gsw_name.clone(),
                descriptor(
                    matrix.clone(),
                    ArtifactConfidentiality::Private,
                    power_rhs_artifact_layout(&secret_metadata, &secret_metadata, role.clone()),
                ),
            );
            let companion_role = crate::rhs::RhsCompanionArtifactRole::RhsCompanion {
                gsw_artifact: gsw_name.clone(),
                source_row: 0,
                target_column: 0,
            };
            artifacts.insert(
                vector_name.clone(),
                descriptor(
                    matrix.clone(),
                    ArtifactConfidentiality::Private,
                    power_encoding_artifact_layout(&sampler, identity, companion_role.clone()),
                ),
            );
            artifacts.insert(
                public_name.clone(),
                descriptor(
                    matrix.clone(),
                    ArtifactConfidentiality::Public,
                    power_encoding_artifact_layout(&sampler, identity, companion_role),
                ),
            );
            package_names.push(PbcSelectorPackageArtifactNames {
                bucket,
                slot,
                package: PowerRhsPackageArtifactNames {
                    gsw_ciphertext: gsw_name,
                    companions: vec![PowerRhsCompanionArtifactName {
                        source_row: 0,
                        target_column: 0,
                        encoding: BggEncodingArtifactNames {
                            vector: vector_name,
                            public_matrix: public_name,
                        },
                    }],
                },
            });
        }
    }
    let names =
        PbcSelectorArtifactNames::canonicalize(layout, key_instance_id, package_names).unwrap();
    let manifest = Manifest {
        ir_version: mxx_ir_core::encoding::IR_VERSION,
        production_id: production_id.clone(),
        artifacts,
    };
    let mut manifest = manifest;
    let package_count = names.selector_packages.len();
    for (role, confidentiality, content_hash) in [
        ("gsw".to_owned(), ArtifactConfidentiality::Private, None),
        ("vector-0".to_owned(), ArtifactConfidentiality::Private, None),
        ("public-0".to_owned(), ArtifactConfidentiality::Public, Some([31; 32])),
    ] {
        manifest.artifacts.insert(
            selector_family_artifact_name_from_names(&layout, &names, key_instance_id, &role),
            ManifestArtifact {
                artifact_type: matrix.clone(),
                family_count: Some(package_count),
                confidentiality,
                content_hash,
                layout: None,
            },
        );
    }
    let gsw_family =
        selector_family_artifact_name_from_names(&layout, &names, key_instance_id, "gsw");
    let imported = PbcSelectorArtifacts::import(
        production_id.clone(),
        layout,
        key_instance_id,
        &manifest,
        names.clone(),
    )
    .unwrap();
    let _initial = BggEncodingWire {
        vector: ring.zero((1, 1)),
        pubkey: BggPublicKeyWire { matrix: ring.zero((1, 1)), reveal_plaintext: false },
        plaintext: None,
    };
    let family_context = DslContext::new("pbc-imported-family-test");
    let selector_families =
        PbcSelectorFamilyInputs::from_artifacts(&ring, layout, &imported, &sampler).unwrap();
    let built = family_context
        .output("gsw", selector_families.gsw().get(Int::constant(0)))
        .unwrap()
        .build()
        .unwrap();
    assert!(built.graph.root_scope().nodes().iter().any(|node| matches!(
        node.kind(),
        mxx_ir_core::node::NodeKind::Input { artifact: Some(_), .. }
    )));
    let validated = built
        .validate_with_manifests(
            &mxx_ir_core::ParamEnv::default(),
            &BTreeMap::from([(production_id.clone(), manifest.clone())]),
        )
        .unwrap();
    let zero = mxx_primitives::matrix::dcrt_poly::DCRTPolyMatrix::zero(&parameters, 1, 1);
    let mut backend = cpu_backend([parameters.clone()]);
    let payload = ArtifactPayload::Matrix(backend.matrix_to_bytes(&zero));
    let family_name = selector_family_artifact_name(&imported, "gsw");
    let mut store = mxx_runtime::artifact::MemoryArtifactStore::default();
    store.store_manifest(manifest.clone()).unwrap();
    for index in 0..package_count {
        store
            .insert(
                ArtifactKey {
                    production: production_id.clone(),
                    name: family_name.clone(),
                    index: Some(index),
                },
                matrix.clone(),
                ArtifactConfidentiality::Private,
                payload.clone(),
            )
            .unwrap();
    }
    let wrong = mxx_primitives::matrix::dcrt_poly::DCRTPolyMatrix::from_poly_vec_row(
        &parameters,
        vec![mxx_primitives::poly::dcrt::poly::DCRTPoly::const_rotate_poly(&parameters, 1)],
    );
    let result = execute(
        &validated,
        &mut backend,
        BTreeMap::from([(
            family_name.clone(),
            RuntimeValue::IndexedFamily(
                (0..package_count).map(|_| RuntimeValue::matrix(wrong.clone())).collect(),
            ),
        )]),
        &mut store,
        SamplingMode::Fresh,
    )
    .unwrap();
    let RuntimeValue::Matrix(value) = &result.outputs["gsw"] else {
        panic!("artifact family output must be a matrix")
    };
    assert_eq!(value.as_ref(), &zero);
    let mut tampered = manifest.clone();
    tampered.artifacts.get_mut(&gsw_family).unwrap().content_hash = Some([1; 32]);
    assert!(
        PbcSelectorArtifacts::import(production_id, layout, key_instance_id, &tampered, names,)
            .is_err()
    );
}
