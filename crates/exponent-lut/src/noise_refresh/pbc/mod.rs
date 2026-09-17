//! Private-bucket cuckoo (PBC) support for sparse-LWR evaluation.
//!
//! PBC separates data that may be public from the secret sparse support. A
//! [`crate::noise_refresh::pbc::PbcPublicLayout`] deterministically hashes every universe
//! coordinate into candidate buckets and rectangularizes those buckets with dummy and padding
//! cells. A [`crate::noise_refresh::pbc::PbcPrivateSchedule`] then records which slot is
//! selected for each bucket to realize the secret support assignment. The
//! compiler consumes the public layout and
//! selector artifacts, while the private schedule is never serialized
//! or used to name public families.
//!
//! The module also exposes clear oracles and diagnostics. They are reference
//! and setup tools, not a replacement for the encrypted DSL graph.

pub(crate) mod artifacts;
mod schedule;

use mxx_dsl::{Family, Mat, Ring};
use mxx_ir_core::artifact::{ArtifactType, Manifest, ProductionId};
use thiserror::Error;

/// Semantic version of the serialized public layout schema.
pub const PBC_LAYOUT_SEMANTIC_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Root seed from which deterministic layout-attempt seeds are derived.
pub struct PbcRootSeed(pub [u8; 32]);

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Seed used for candidate-bucket hashing in one layout attempt.
pub struct PbcLayoutSeed(pub [u8; 32]);

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Public location of a coordinate replica in the rectangular layout.
pub struct PbcLocation {
    /// Bucket row containing the cell.
    pub bucket: usize,
    /// Slot within that bucket row.
    pub slot: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// A public cell used by the bucket compiler.
pub enum PbcCell {
    /// A universe coordinate and its hash replica.
    Real {
        /// Coordinate in the original public universe.
        coordinate: usize,
        /// Candidate replica number for that coordinate.
        replica: usize,
    },
    /// A harmless public zero cell used when a bucket has no selected support.
    Dummy,
    /// Storage-only cell added to make every bucket the same width.
    Padding,
}

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
/// Digest of the complete public PBC layout and its parameters.
pub struct PbcLayoutId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Public reason why a layout seed attempt was rejected.
pub enum PbcRetryCause {
    /// A bucket exceeded the configured rectangular-width limit.
    BucketWidthExceeded,
    /// The secret support could not be assigned to distinct candidate buckets.
    NoPerfectSchedule,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// Aggregate diagnostics for failed or accepted public layout attempts.
pub struct PbcRetryDiagnostics {
    /// Number of attempts performed.
    pub attempts: u32,
    /// Attempts rejected for excessive bucket width.
    pub bucket_width_failures: u32,
    /// Attempts rejected because matching found no schedule.
    pub no_perfect_schedule_failures: u32,
    /// Last failure cause, if no attempt was accepted.
    pub last_public_cause: Option<PbcRetryCause>,
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
/// Errors from PBC parameter, layout, schedule, or artifact validation.
pub enum PbcError {
    #[error("invalid PBC parameters: {0}")]
    /// Parameter validation failed.
    InvalidParameters(String),
    #[error("sparse support has the wrong size")]
    /// Support cardinality differs from the configured weight.
    SupportSize,
    #[error("sparse support contains a duplicate or out-of-range coordinate")]
    /// Support contains an invalid coordinate or duplicate.
    InvalidSupport,
    #[error("candidate derivation nonce overflow")]
    /// Hash nonce space was exhausted.
    HashNonceOverflow,
    #[error("derived bucket width exceeds the configured limit")]
    /// Rectangularization would exceed the configured width limit.
    BucketWidthExceeded,
    #[error("the cuckoo graph has no schedule covering the support")]
    /// No distinct bucket assignment covers the support.
    NoPerfectSchedule,
    #[error("all PBC seed attempts failed: {0:?}")]
    /// Retry budget was exhausted.
    SeedAttemptsExhausted(PbcRetryDiagnostics),
    #[error("public PBC layout is malformed: {0}")]
    /// Public layout structure or identity is invalid.
    InvalidLayout(String),
    #[error("private PBC schedule is malformed: {0}")]
    /// Private schedule does not match its public layout.
    InvalidSchedule(String),
    #[error("PBC layout identity mismatch")]
    /// Values were paired with a different layout digest.
    LayoutIdentityMismatch,
    #[error("artifact layout or key identity mismatch")]
    /// Artifact names or metadata do not match the requested setup.
    ArtifactIdentityMismatch,
    #[error("integer conversion or size overflow")]
    /// A host-size conversion or arithmetic bound overflowed.
    SizeOverflow,
}

pub use artifacts::{
    PbcSelectorArtifactNames, PbcSelectorArtifacts, PbcSelectorFamilyMetadata,
    PbcSelectorPackageArtifactNames, PbcSelectorRuntimeBinding, PbcTrustedSelectorBits,
    canonical_component_name, public_family_artifact_name, selector_bit_family_name,
    selector_family_artifact_name, selector_family_artifact_name_from_names,
};
pub use schedule::{
    BinaryTransformMatrix, PbcGeneratedKeyLayout, PbcPrivateSchedule, canonical_decode,
    clear_pbc_inner_product, dense_binary_support, generate_key_layout, support_from_dense,
};

#[cfg(test)]
mod tests {
    use super::{
        schedule::{ValidatedSupport, deterministic_matching, schedule, schedule_from_owners},
        *,
    };

    fn seed(byte: u8) -> PbcRootSeed {
        PbcRootSeed([byte; 32])
    }

    fn has_matching(layout: &PbcPublicLayout, support: &[usize]) -> bool {
        fn visit(
            layout: &PbcPublicLayout,
            support: &[usize],
            index: usize,
            used: &mut [bool],
        ) -> bool {
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
    fn balanced_product_schedule_handles_identity_singleton_and_odd_levels() {
        let empty = BalancedProductSchedule::new(0).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.level_count(), 0);
        assert!(empty.children(1, 0).is_none());

        let singleton = BalancedProductSchedule::new(1).unwrap();
        assert_eq!(singleton.level_count(), 1);
        assert_eq!(singleton.level_width(0), Some(1));
        assert!(singleton.children(1, 0).is_none());

        let odd = BalancedProductSchedule::new(5).unwrap();
        assert_eq!(odd.level_count(), 4);
        assert_eq!(odd.level_width(0), Some(5));
        assert_eq!(odd.level_width(1), Some(3));
        assert_eq!(odd.level_width(2), Some(2));
        assert_eq!(odd.level_width(3), Some(1));
        assert_eq!(odd.children(1, 0), Some((0, Some(1))));
        assert_eq!(odd.children(1, 2), Some((4, None)));
        assert_eq!(odd.children(2, 1), Some((2, None)));
        assert!(odd.children(4, 0).is_none());
        assert!(odd.validate().is_ok());
    }

    #[test]
    fn balanced_product_schedule_rejects_noncanonical_serialized_levels() {
        let mut schedule = BalancedProductSchedule::new(3).unwrap();
        schedule.level_widths[1] = 3;
        assert!(schedule.validate().is_err());
        let encoded = serde_json::json!({"leaf_count": 3, "level_widths": [3, 3, 1]});
        let decoded = serde_json::from_value::<BalancedProductSchedule>(encoded).unwrap();
        assert!(decoded.validate().is_err());
    }

    #[test]
    fn named_profile_numeric_fields_are_bound_and_deserialization_is_validated() {
        for parameters in
            [PbcParameters::conservative(10, 4), PbcParameters::paper_evaluation(10, 4)]
        {
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
                assert!(
                    PbcPublicLayout::build(&forged, derive_attempt_seed(seed(20), 0), 0).is_err()
                );

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
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(1), 0), 0).unwrap();
        layout.validate().unwrap();
        assert_eq!(layout.cells.len(), parameters.bucket_count);
        assert!(layout.cells.iter().all(|row| row.len() == layout.bucket_width));
        assert_eq!(
            layout
                .cells
                .iter()
                .flatten()
                .filter(|cell| matches!(cell, PbcCell::Real { .. }))
                .count(),
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
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(41), 0), 0).unwrap();
        let (padding_bucket, padding_slot) = layout
            .cells
            .iter()
            .enumerate()
            .find_map(|(bucket, row)| {
                row.iter()
                    .position(|cell| matches!(cell, PbcCell::Padding))
                    .map(|slot| (bucket, slot))
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
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(2), 0), 0).unwrap();
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
                        assert_eq!(
                            canonical_decode(&layout, &scheduled).unwrap(),
                            support.to_vec()
                        );
                    }
                    assert_eq!(
                        schedule(&layout, &validated).is_ok(),
                        expected,
                        "support={support:?}"
                    );
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
                    assert_eq!(
                        sample.accepted_attempt,
                        Some(generated.public_layout.accepted_attempt)
                    );
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
                        generate_key_layout(&parameters, seed(seed_byte), &[first, second])
                            .unwrap();
                    assert!(
                        generated
                            .private_schedule()
                            .assigned_coordinates
                            .iter()
                            .any(Option::is_none),
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
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(10), 0), 0).unwrap();
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
        assert_ne!(a.identity(), b.identity());
        let a_binding = PbcPublicVectorFamilyBinding::from_encoded(&layout, &a).unwrap();
        let b_binding = PbcPublicVectorFamilyBinding::from_encoded(&layout, &b).unwrap();
        assert_ne!(a_binding.identity(), b_binding.identity());
        // Selector/setup names are D0-independent, while encrypt-stage
        // public-factor names are content-bound to the supplied D0.
        let selector_a = PbcSelectorArtifactNames::canonicalize_schema(&layout, [3; 32]).unwrap();
        let selector_b = PbcSelectorArtifactNames::canonicalize_schema(&layout, [3; 32]).unwrap();
        assert_eq!(selector_a, selector_b);
        assert_ne!(public_family_artifact_name(&a, "d0"), public_family_artifact_name(&b, "d0"));
    }

    #[test]
    fn selector_names_are_layout_and_key_namespace_bound_but_schedule_independent() {
        let parameters = PbcParameters::custom(7, 2, 2, 3, 1, None);
        let layout =
            PbcPublicLayout::build(&parameters, derive_attempt_seed(seed(11), 0), 0).unwrap();
        let package_names = |key_instance_id| {
            layout
                .cells
                .iter()
                .enumerate()
                .flat_map(|(bucket, row)| {
                    row.iter().enumerate().filter_map(move |(slot, cell)| {
                        (!matches!(cell, PbcCell::Padding)).then_some(
                            PbcSelectorPackageArtifactNames {
                                bucket,
                                slot,
                                ciphertext: canonical_component_name(
                                    layout.layout_id,
                                    key_instance_id,
                                    bucket,
                                    slot,
                                ),
                            },
                        )
                    })
                })
                .collect::<Vec<_>>()
        };
        let names_a =
            PbcSelectorArtifactNames::canonicalize(&layout, [1; 32], package_names([1; 32]))
                .unwrap();
        let names_b =
            PbcSelectorArtifactNames::canonicalize(&layout, [2; 32], package_names([2; 32]))
                .unwrap();
        assert_eq!(
            names_a.selector_packages.len(),
            parameters.universe_size * parameters.hash_count + parameters.bucket_count
        );
        assert_ne!(names_a, names_b);
        assert!(
            names_a.selector_packages.iter().all(|entry| !entry.ciphertext.contains("selected"))
        );
        assert!(
            names_a.selector_packages.iter().all(|entry| !entry.ciphertext.contains("support"))
        );
    }

    #[test]
    fn trusted_selector_bits_validate_count_order_and_binary_values() {
        let parameters = PbcParameters::custom(5, 1, 2, 3, 1, None);
        let generated = generate_key_layout(&parameters, seed(15), &[0]).unwrap();
        let layout = &generated.public_layout;
        let ring = Ring::new(257, 4);
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
            active.iter().all(|(bucket, slot, _)| {
                !matches!(layout.cells[bucket][slot], PbcCell::Padding)
            })
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
                *bucket != first_bucket &&
                    generated.private_schedule().selected_slot(*bucket) != *slot
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
        assert_eq!(trusted.runtime_bits(), bits);
        assert_eq!(trusted.family().count(), &mxx_ir_core::IntExpr::constant(active.len()));
        assert!(!trusted.input_name().contains("support"));
    }

    #[test]
    fn trusted_selector_bits_bind_through_the_runtime_family_input() {
        use mxx_dsl::DslContext;
        use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
        use mxx_runtime::{
            Backend, MemoryArtifactStore, RuntimeValue, backend::poly::cpu_backend, execute,
            transcript::SamplingMode,
        };
        use num_bigint::{BigInt, Sign};

        let parameters = PbcParameters::custom(5, 1, 2, 3, 1, None);
        let generated = generate_key_layout(&parameters, seed(18), &[0]).unwrap();
        let poly_parameters = DCRTPolyParams::new(4, 1, 20, 2, None, None);
        let modulus = BigInt::from_biguint(Sign::Plus, poly_parameters.modulus().as_ref().clone());
        let ring = mxx_dsl::Ring::new(modulus, 4);
        let trusted = PbcTrustedSelectorBits::from_schedule(&generated, &ring, [9; 32]).unwrap();
        let runtime_binding = trusted.runtime_binding(&ring).unwrap();
        let materialization_count = std::cell::Cell::new(0usize);
        let oracle_members = runtime_binding
            .materialize_members(|bit| {
                materialization_count.set(materialization_count.get() + 1);
                Ok::<_, std::convert::Infallible>(bit)
            })
            .unwrap();
        assert_eq!(materialization_count.get(), 2);
        assert_eq!(oracle_members, trusted.runtime_bits());
        let graph = DslContext::new("trusted-selector-runtime-binding")
            .artifact_output("bits", trusted.family().clone())
            .unwrap()
            .build()
            .unwrap()
            .validate(&Default::default())
            .unwrap();
        let mut backend = cpu_backend([poly_parameters]);
        let inputs = runtime_binding.materialize(&mut backend).unwrap();
        let result = execute(
            &graph,
            &mut backend,
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        let RuntimeValue::IndexedFamily(actual) = &result.outputs["bits"] else {
            panic!("selector output must remain a runtime family")
        };
        assert_eq!(actual.len(), trusted.runtime_bits().len());
        let scalar_type = artifacts::concrete_matrix_type(&ring, (1, 1)).unwrap();
        for (actual, bit) in actual.iter().zip(trusted.runtime_bits()) {
            let RuntimeValue::Matrix(actual) = actual else {
                panic!("selector member must be a matrix")
            };
            let expected = backend
                .constant_matrix(
                    &scalar_type,
                    &mxx_ir_core::node::ConstantMatrix::Polynomial {
                        coefficients: vec![mxx_ir_core::IntExpr::constant(*bit)],
                    },
                    &Default::default(),
                )
                .unwrap();
            assert_eq!(actual.as_ref(), &expected);
        }
    }
}

use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Named parameter policy used when generating a PBC layout.
pub enum PbcProfile {
    /// A conservative bucket-count and retry policy.
    Conservative,
    /// Parameters using three candidate buckets per coordinate and a larger
    /// deterministic retry budget.
    PaperEvaluation,
    /// Explicit values supplied through [`PbcParameters::custom`].
    Custom,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// Validated public dimensions and retry limits for PBC.
pub struct PbcParameters {
    /// Number of coordinates in the public universe.
    pub universe_size: usize,
    /// Number of nonzero coordinates in the sparse support.
    pub support_weight: usize,
    /// Number of candidate buckets hashed for each coordinate.
    pub hash_count: usize,
    /// Number of bucket rows in the rectangular layout.
    pub bucket_count: usize,
    /// Maximum number of deterministic seeds tried by schedule generation.
    pub max_seed_attempts: u32,
    /// Optional hard limit on the rectangular bucket width.
    pub bucket_width_limit: Option<usize>,
    /// Preset or custom policy associated with these values.
    pub profile: PbcProfile,
}

#[derive(Deserialize)]
struct PbcParametersRepr {
    universe_size: usize,
    support_weight: usize,
    hash_count: usize,
    bucket_count: usize,
    max_seed_attempts: u32,
    bucket_width_limit: Option<usize>,
    profile: PbcProfile,
}

impl<'de> Deserialize<'de> for PbcParameters {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PbcParametersRepr::deserialize(deserializer)?;
        let parameters = Self {
            universe_size: repr.universe_size,
            support_weight: repr.support_weight,
            hash_count: repr.hash_count,
            bucket_count: repr.bucket_count,
            max_seed_attempts: repr.max_seed_attempts,
            bucket_width_limit: repr.bucket_width_limit,
            profile: repr.profile,
        };
        parameters.validate().map_err(serde::de::Error::custom)?;
        Ok(parameters)
    }
}

impl PbcParameters {
    /// Creates the conservative preset for a universe and support weight.
    pub fn conservative(universe_size: usize, support_weight: usize) -> Self {
        Self {
            universe_size,
            support_weight,
            hash_count: 3,
            bucket_count: support_weight.saturating_mul(3).div_ceil(2).max(3),
            max_seed_attempts: 16,
            bucket_width_limit: None,
            profile: PbcProfile::Conservative,
        }
    }

    /// Creates the numeric policy with three candidate buckets per coordinate,
    /// `bucket_count = max(3, support_weight + 3)`, and 128 retries.
    pub fn paper_evaluation(universe_size: usize, support_weight: usize) -> Self {
        Self {
            universe_size,
            support_weight,
            hash_count: 3,
            bucket_count: support_weight.saturating_add(3).max(3),
            max_seed_attempts: 128,
            bucket_width_limit: None,
            profile: PbcProfile::PaperEvaluation,
        }
    }

    /// Creates an explicit parameter set; call [`Self::validate`] before use.
    pub fn custom(
        universe_size: usize,
        support_weight: usize,
        hash_count: usize,
        bucket_count: usize,
        max_seed_attempts: u32,
        bucket_width_limit: Option<usize>,
    ) -> Self {
        Self {
            universe_size,
            support_weight,
            hash_count,
            bucket_count,
            max_seed_attempts,
            bucket_width_limit,
            profile: PbcProfile::Custom,
        }
    }

    /// Checks dimensions, overflow limits, and profile-specific invariants.
    pub fn validate(&self) -> Result<(), PbcError> {
        if self.universe_size == 0 {
            return Err(PbcError::InvalidParameters("universe_size must be positive".into()));
        }
        if self.support_weight == 0 || self.support_weight > self.universe_size {
            return Err(PbcError::InvalidParameters(
                "support_weight must be in 1..=universe_size".into(),
            ));
        }
        if self.hash_count < 2 {
            return Err(PbcError::InvalidParameters("hash_count must be at least two".into()));
        }
        if self.bucket_count < self.support_weight {
            return Err(PbcError::InvalidParameters("bucket_count must cover the support".into()));
        }
        if self.bucket_count < self.hash_count {
            return Err(PbcError::InvalidParameters(
                "bucket_count must be at least hash_count".into(),
            ));
        }
        if self.max_seed_attempts == 0 {
            return Err(PbcError::InvalidParameters("max_seed_attempts must be positive".into()));
        }
        if self.bucket_width_limit.is_some_and(|width| width < 2) {
            return Err(PbcError::InvalidParameters(
                "bucket_width_limit must be at least two".into(),
            ));
        }
        for value in [self.universe_size, self.support_weight, self.hash_count, self.bucket_count] {
            if u64::try_from(value).is_err() {
                return Err(PbcError::SizeOverflow);
            }
        }
        match self.profile {
            PbcProfile::Conservative => {
                let expected_bucket_count = self
                    .support_weight
                    .checked_mul(3)
                    .ok_or(PbcError::SizeOverflow)?
                    .div_ceil(2)
                    .max(3);
                if self.hash_count != 3 ||
                    self.bucket_count != expected_bucket_count ||
                    self.max_seed_attempts != 16
                {
                    return Err(PbcError::InvalidParameters(
                        "Conservative profile requires hash_count=3, bucket_count=max(3, ceil(3*support_weight/2)), and max_seed_attempts=16".into(),
                    ));
                }
            }
            PbcProfile::PaperEvaluation => {
                let expected_bucket_count =
                    self.support_weight.checked_add(3).ok_or(PbcError::SizeOverflow)?.max(3);
                if self.hash_count != 3 ||
                    self.bucket_count != expected_bucket_count ||
                    self.max_seed_attempts != 128
                {
                    return Err(PbcError::InvalidParameters(
                        "PaperEvaluation profile requires hash_count=3, bucket_count=max(3, support_weight+3), and max_seed_attempts=128".into(),
                    ));
                }
            }
            PbcProfile::Custom => {}
        }
        Ok(())
    }

    /// Appends the canonical field order and profile tag used in layout IDs.
    ///
    /// This is an identity encoding, not a general-purpose serialization:
    /// callers validate the parameters first and then include these fields in
    /// the public layout digest.
    pub(crate) fn encode_canonical(&self, out: &mut Vec<u8>) -> Result<(), PbcError> {
        for value in [self.universe_size, self.support_weight, self.hash_count, self.bucket_count] {
            out.extend(u64::try_from(value).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        }
        out.extend(u64::from(self.max_seed_attempts).to_le_bytes());
        match self.bucket_width_limit {
            Some(width) => {
                out.push(1);
                out.extend(u64::try_from(width).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
            }
            None => out.push(0),
        }
        out.push(match self.profile {
            PbcProfile::Conservative => 0,
            PbcProfile::PaperEvaluation => 1,
            PbcProfile::Custom => 2,
        });
        Ok(())
    }
}

/// Derives the public seed for one retry attempt.
pub fn derive_attempt_seed(root_seed: PbcRootSeed, attempt: u32) -> PbcLayoutSeed {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-exponent-lut/pbc/layout-seed/v1");
    hasher.update(root_seed.0);
    hasher.update(u64::from(attempt).to_le_bytes());
    PbcLayoutSeed(hasher.finalize().into())
}

/// Derives `hash_count` distinct candidate buckets for one coordinate.
///
/// The replica number is part of the hash domain, so each candidate has a
/// stable position in the public candidate list. Rejection sampling accepts
/// only the largest multiple of `bucket_count` below `2^64`; the accepted
/// digest is then reduced without modulo bias.
pub fn derive_candidate_buckets(
    seed: PbcLayoutSeed,
    coordinate: usize,
    bucket_count: usize,
    hash_count: usize,
) -> Result<Vec<usize>, PbcError> {
    if bucket_count == 0 || hash_count == 0 || hash_count > bucket_count {
        return Err(PbcError::InvalidParameters(
            "candidate derivation requires 0 < hash_count <= bucket_count".into(),
        ));
    }
    let coordinate = u64::try_from(coordinate).map_err(|_| PbcError::SizeOverflow)?;
    let bucket_count = u64::try_from(bucket_count).map_err(|_| PbcError::SizeOverflow)?;
    let range = 1u128 << 64;
    let limit = (range / u128::from(bucket_count)) * u128::from(bucket_count);
    let mut candidates = Vec::with_capacity(hash_count);
    for replica in 0..hash_count {
        let replica = u64::try_from(replica).map_err(|_| PbcError::SizeOverflow)?;
        let mut nonce = 0u64;
        loop {
            let mut hasher = Sha256::new();
            hasher.update(b"mxx-exponent-lut/pbc/candidate/v1");
            hasher.update(seed.0);
            hasher.update(coordinate.to_le_bytes());
            hasher.update(replica.to_le_bytes());
            hasher.update(nonce.to_le_bytes());
            let digest = hasher.finalize();
            let z = u64::from_le_bytes(
                digest[..8].try_into().map_err(|_| PbcError::HashNonceOverflow)?,
            );
            let z128 = u128::from(z);
            if z128 < limit {
                let bucket = (z128 % u128::from(bucket_count)) as usize;
                if !candidates.contains(&bucket) {
                    candidates.push(bucket);
                    break;
                }
            }
            nonce = nonce.checked_add(1).ok_or(PbcError::HashNonceOverflow)?;
        }
    }
    Ok(candidates)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
/// Public, serializable description of all PBC buckets and coordinate replicas.
pub struct PbcPublicLayout {
    /// Layout schema version checked during deserialization.
    pub semantic_version: u32,
    /// Public dimensions and retry policy used to build the layout.
    pub parameters: PbcParameters,
    /// Accepted attempt seed.
    pub seed: PbcLayoutSeed,
    /// Zero-based attempt number at which this layout was accepted.
    pub accepted_attempt: u32,
    /// Candidate bucket list for each universe coordinate.
    pub candidates: Vec<Vec<usize>>,
    /// Rectangular bucket rows containing real, dummy, and padding cells.
    pub cells: Vec<Vec<PbcCell>>,
    /// Inverse coordinate-to-replica locations.
    pub locations: Vec<Vec<PbcLocation>>,
    /// Common row width after rectangularization.
    pub bucket_width: usize,
    /// Digest binding all public layout fields.
    pub layout_id: PbcLayoutId,
}

#[derive(Deserialize)]
struct PbcPublicLayoutRepr {
    semantic_version: u32,
    parameters: PbcParameters,
    seed: PbcLayoutSeed,
    accepted_attempt: u32,
    candidates: Vec<Vec<usize>>,
    cells: Vec<Vec<PbcCell>>,
    locations: Vec<Vec<PbcLocation>>,
    bucket_width: usize,
    layout_id: PbcLayoutId,
}

impl<'de> Deserialize<'de> for PbcPublicLayout {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = PbcPublicLayoutRepr::deserialize(deserializer)?;
        let layout = Self {
            semantic_version: repr.semantic_version,
            parameters: repr.parameters,
            seed: repr.seed,
            accepted_attempt: repr.accepted_attempt,
            candidates: repr.candidates,
            cells: repr.cells,
            locations: repr.locations,
            bucket_width: repr.bucket_width,
            layout_id: repr.layout_id,
        };
        layout.validate().map_err(serde::de::Error::custom)?;
        Ok(layout)
    }
}

impl PbcPublicLayout {
    /// Builds and validates a public layout for one accepted seed attempt.
    pub fn build(
        parameters: &PbcParameters,
        seed: PbcLayoutSeed,
        accepted_attempt: u32,
    ) -> Result<Self, PbcError> {
        parameters.validate()?;
        if accepted_attempt >= parameters.max_seed_attempts {
            return Err(PbcError::InvalidParameters(
                "accepted attempt exceeds max_seed_attempts".into(),
            ));
        }
        let mut candidates = Vec::with_capacity(parameters.universe_size);
        let mut bucket_real_cells = vec![Vec::<PbcCell>::new(); parameters.bucket_count];
        for coordinate in 0..parameters.universe_size {
            // `candidate_buckets[x]` is the ordered public hash-neighborhood
            // of x; each replica is inserted into exactly one bucket row.
            let candidate_buckets = derive_candidate_buckets(
                seed,
                coordinate,
                parameters.bucket_count,
                parameters.hash_count,
            )?;
            for (replica, &bucket) in candidate_buckets.iter().enumerate() {
                bucket_real_cells[bucket].push(PbcCell::Real { coordinate, replica });
            }
            candidates.push(candidate_buckets);
        }
        for cells in &mut bucket_real_cells {
            cells.sort_by_key(|cell| match cell {
                PbcCell::Real { coordinate, replica } => (*coordinate, *replica),
                PbcCell::Dummy | PbcCell::Padding => (usize::MAX, usize::MAX),
            });
        }
        // Rectangularization adds one dummy to every row, then enough padding
        // for the row with the most real replicas.
        let max_real = bucket_real_cells.iter().map(Vec::len).max().unwrap_or(0);
        let bucket_width = max_real.checked_add(1).ok_or(PbcError::SizeOverflow)?;
        if parameters.bucket_width_limit.is_some_and(|limit| bucket_width > limit) {
            return Err(PbcError::BucketWidthExceeded);
        }
        let cells = bucket_real_cells
            .into_iter()
            .map(|mut real| {
                real.push(PbcCell::Dummy);
                real.resize(bucket_width, PbcCell::Padding);
                real
            })
            .collect::<Vec<_>>();
        // `locations[x][r]` is the inverse address used by private matching;
        // it is public because it reveals only the hash layout, not support.
        let mut locations = vec![
            vec![PbcLocation { bucket: 0, slot: 0 }; parameters.hash_count];
            parameters.universe_size
        ];
        for (bucket, row) in cells.iter().enumerate() {
            for (slot, cell) in row.iter().enumerate() {
                if let PbcCell::Real { coordinate, replica } = cell {
                    locations[*coordinate][*replica] = PbcLocation { bucket, slot };
                }
            }
        }
        let layout = Self {
            semantic_version: PBC_LAYOUT_SEMANTIC_VERSION,
            parameters: parameters.clone(),
            seed,
            accepted_attempt,
            candidates,
            cells,
            locations,
            bucket_width,
            layout_id: PbcLayoutId([0; 32]),
        };
        let layout_id = layout.compute_id()?;
        let layout = Self { layout_id, ..layout };
        layout.validate()?;
        Ok(layout)
    }

    /// Recomputes all structural invariants and the layout identity.
    pub fn validate(&self) -> Result<(), PbcError> {
        self.parameters.validate()?;
        if self.semantic_version != PBC_LAYOUT_SEMANTIC_VERSION {
            return Err(PbcError::InvalidLayout("unsupported PBC layout semantic version".into()));
        }
        if self.accepted_attempt >= self.parameters.max_seed_attempts {
            return Err(PbcError::InvalidLayout("accepted attempt is out of range".into()));
        }
        if self.candidates.len() != self.parameters.universe_size {
            return Err(PbcError::InvalidLayout("candidate row count mismatch".into()));
        }
        let mut expected_real_count = vec![0usize; self.parameters.bucket_count];
        for (coordinate, row) in self.candidates.iter().enumerate() {
            if row.len() != self.parameters.hash_count {
                return Err(PbcError::InvalidLayout("candidate count mismatch".into()));
            }
            let mut seen = HashSet::with_capacity(row.len());
            for &bucket in row {
                if bucket >= self.parameters.bucket_count || !seen.insert(bucket) {
                    return Err(PbcError::InvalidLayout("candidate bucket is invalid".into()));
                }
                expected_real_count[bucket] += 1;
            }
            let derived = derive_candidate_buckets(
                self.seed,
                coordinate,
                self.parameters.bucket_count,
                self.parameters.hash_count,
            )?;
            if derived != *row {
                return Err(PbcError::InvalidLayout("candidate derivation mismatch".into()));
            }
        }
        if self.cells.len() != self.parameters.bucket_count || self.bucket_width == 0 {
            return Err(PbcError::InvalidLayout("bucket row count or width mismatch".into()));
        }
        let mut observed_real = vec![0usize; self.parameters.bucket_count];
        let mut seen_real = HashSet::with_capacity(
            self.parameters
                .universe_size
                .checked_mul(self.parameters.hash_count)
                .ok_or(PbcError::SizeOverflow)?,
        );
        for (bucket, row) in self.cells.iter().enumerate() {
            if row.len() != self.bucket_width {
                return Err(PbcError::InvalidLayout("bucket is not rectangular".into()));
            }
            let dummy_slot = row
                .iter()
                .position(|cell| matches!(cell, PbcCell::Dummy))
                .ok_or_else(|| PbcError::InvalidLayout("missing dummy".into()))?;
            if row.iter().filter(|cell| matches!(cell, PbcCell::Dummy)).count() != 1 {
                return Err(PbcError::InvalidLayout("bucket must contain one dummy".into()));
            }
            let mut previous_real = None;
            for (slot, cell) in row.iter().enumerate() {
                match cell {
                    PbcCell::Real { coordinate, replica } => {
                        if *coordinate >= self.parameters.universe_size ||
                            *replica >= self.parameters.hash_count ||
                            slot > dummy_slot
                        {
                            return Err(PbcError::InvalidLayout("invalid real cell".into()));
                        }
                        if let Some(previous) = previous_real &&
                            previous >= (*coordinate, *replica)
                        {
                            return Err(PbcError::InvalidLayout(
                                "real cells are not canonical".into(),
                            ));
                        }
                        previous_real = Some((*coordinate, *replica));
                        if self.candidates[*coordinate][*replica] != bucket ||
                            !seen_real.insert((*coordinate, *replica))
                        {
                            return Err(PbcError::InvalidLayout(
                                "real cell is not canonical".into(),
                            ));
                        }
                        observed_real[bucket] += 1;
                    }
                    PbcCell::Dummy if slot != dummy_slot => {
                        return Err(PbcError::InvalidLayout("dummy ordering is invalid".into()));
                    }
                    PbcCell::Padding if slot <= dummy_slot => {
                        return Err(PbcError::InvalidLayout("padding must follow dummy".into()));
                    }
                    PbcCell::Dummy | PbcCell::Padding => {}
                }
            }
        }
        if observed_real != expected_real_count ||
            seen_real.len() !=
                self.parameters
                    .universe_size
                    .checked_mul(self.parameters.hash_count)
                    .ok_or(PbcError::SizeOverflow)? ||
            self.bucket_width != 1 + expected_real_count.iter().copied().max().unwrap_or(0)
        {
            return Err(PbcError::InvalidLayout("real-cell accounting mismatch".into()));
        }
        if self.locations.len() != self.parameters.universe_size ||
            self.locations.iter().any(|row| row.len() != self.parameters.hash_count)
        {
            return Err(PbcError::InvalidLayout("location map shape mismatch".into()));
        }
        for (coordinate, row) in self.locations.iter().enumerate() {
            for (replica, location) in row.iter().enumerate() {
                if location.bucket >= self.parameters.bucket_count ||
                    location.slot >= self.bucket_width ||
                    !matches!(
                        self.cells[location.bucket][location.slot],
                        PbcCell::Real { coordinate: x, replica: r }
                            if x == coordinate && r == replica
                    )
                {
                    return Err(PbcError::InvalidLayout("location inverse map mismatch".into()));
                }
            }
        }
        if self.compute_id()? != self.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        Ok(())
    }

    /// Commits the public `T_l` label-transform order together with the
    /// balanced product shape used to lower it.  The matrices themselves are
    /// runtime public-family material and are intentionally not copied into
    /// setup metadata; the layout identity commits their canonical cell and
    /// replica order.
    pub fn label_transform_commitment(
        &self,
        schedule: &BalancedProductSchedule,
    ) -> Result<[u8; 32], PbcError> {
        self.validate()?;
        schedule.validate()?;
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/sparse-lwr/t-l-commitment/v1");
        digest.update(self.layout_id.0);
        digest.update(serde_json::to_vec(schedule).map_err(|_| {
            PbcError::InvalidLayout("balanced product schedule is not serializable".into())
        })?);
        Ok(digest.finalize().into())
    }

    /// Commits caller-supplied preprocessing transforms together with their
    /// balanced product shape.  This is the identity used when a setup uses
    /// nontrivial binary `T_l` matrices instead of the canonical one-hot PBC
    /// projections.
    pub fn label_transform_commitment_with_matrices(
        &self,
        schedule: &BalancedProductSchedule,
        transforms: &[BinaryTransformMatrix],
    ) -> Result<[u8; 32], PbcError> {
        self.validate()?;
        schedule.validate()?;
        if transforms.is_empty() {
            return Err(PbcError::InvalidLayout("label transform schedule is empty".into()));
        }
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/sparse-lwr/t-l-matrices/v1");
        digest.update(self.layout_id.0);
        digest.update(serde_json::to_vec(schedule).map_err(|_| {
            PbcError::InvalidLayout("balanced product schedule is not serializable".into())
        })?);
        for transform in transforms {
            digest.update(transform.commitment());
        }
        Ok(digest.finalize().into())
    }

    fn compute_id(&self) -> Result<PbcLayoutId, PbcError> {
        let mut encoding = Vec::new();
        encoding.extend(u64::from(self.semantic_version).to_le_bytes());
        self.parameters.encode_canonical(&mut encoding)?;
        encoding.extend(self.seed.0);
        encoding.extend(u64::from(self.accepted_attempt).to_le_bytes());
        encode_nested_usizes(&mut encoding, &self.candidates)?;
        encode_cells(&mut encoding, &self.cells)?;
        encoding.extend(
            u64::try_from(self.bucket_width).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes(),
        );
        let mut hasher = Sha256::new();
        hasher.update(b"mxx-exponent-lut/pbc/layout-id/v1");
        hasher.update(encoding);
        Ok(PbcLayoutId(hasher.finalize().into()))
    }
}

/// The canonical row-major index of all cells that can carry a selector.
///
/// A PBC row contains real cells followed by exactly one dummy cell and then
/// optional padding cells.  This index includes the real and dummy cells, but
/// never padding.  The third tuple component is a dense index over those
/// active cells, so a flattened selector family can use it without scanning
/// the rectangular storage again.
///
/// The index contains derived, non-serialized state.  Construct it with
/// [`PbcActiveCellIndex::build`] after obtaining a validated public layout;
/// construction validates the layout itself and therefore cannot accidentally
/// index malformed or untrusted serialized data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PbcActiveCellIndex {
    /// `(bucket, slot, flat_nonpadding_index)` in canonical row-major order.
    entries: Vec<(usize, usize, usize)>,
    /// Half-open entry ranges for each bucket, with one final total offset.
    bucket_offsets: Vec<usize>,
}

/// Shape of a balanced product reduction.
///
/// This is deliberately only a compact description of the reduction shape;
/// it does not contain one node (or one host value) per member.  Evaluators
/// use the widths and [`Self::children`] while lowering their one structural
/// family body.  In particular, an odd level carries its final singleton to
/// the next level instead of multiplying it by an identity value.  An empty
/// product is represented by a zero-leaf schedule and is handled by the
/// evaluator's identity wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BalancedProductSchedule {
    leaf_count: usize,
    level_widths: Vec<usize>,
}

impl BalancedProductSchedule {
    /// Builds a schedule in `O(log(leaf_count))` host work.
    pub fn new(leaf_count: usize) -> Result<Self, PbcError> {
        let mut level_widths = Vec::new();
        let mut width = leaf_count;
        while width != 0 {
            level_widths.push(width);
            if width == 1 {
                break;
            }
            width = width.div_ceil(2);
        }
        // Keep the public representation canonical even if this type is later
        // constructed from a deserialized manifest.
        if leaf_count != 0 && level_widths.last() != Some(&1) {
            return Err(PbcError::InvalidLayout("balanced product root is missing".into()));
        }
        Ok(Self { leaf_count, level_widths })
    }

    /// Number of leaves in the input product. Zero denotes the identity.
    pub const fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Returns true when the product is the empty identity product.
    pub const fn is_empty(&self) -> bool {
        self.leaf_count == 0
    }

    /// Returns the number of reduction levels, including the leaf level.
    pub fn level_count(&self) -> usize {
        self.level_widths.len()
    }

    /// Returns the number of values present at `level` (`0` is the leaf
    /// level). An out-of-range level is rejected instead of silently creating
    /// a malformed family range.
    pub fn level_width(&self, level: usize) -> Option<usize> {
        self.level_widths.get(level).copied()
    }

    /// Returns the input positions consumed by one parent at `level`.
    ///
    /// `level` names the parent level, so `level == 1` consumes leaves. The
    /// right child is `None` for a carried singleton. A singleton schedule
    /// has no parent level and therefore returns `None`.
    pub fn children(&self, level: usize, parent: usize) -> Option<(usize, Option<usize>)> {
        let parent_width = *self.level_widths.get(level)?;
        if level == 0 || parent >= parent_width {
            return None;
        }
        let child_width = self.level_widths[level - 1];
        let left = parent.checked_mul(2)?;
        if left >= child_width {
            return None;
        }
        let right = (left + 1 < child_width).then_some(left + 1);
        Some((left, right))
    }

    /// Returns the canonical leaf count for a serialized schedule.
    ///
    /// This check is intentionally strict: stage identities must not accept a
    /// hand-edited level vector that changes singleton reuse or tree depth.
    pub fn validate(&self) -> Result<(), PbcError> {
        let expected = Self::new(self.leaf_count)?;
        if self.level_widths != expected.level_widths {
            return Err(PbcError::InvalidLayout(
                "balanced product schedule levels are not canonical".into(),
            ));
        }
        Ok(())
    }
}

impl PbcActiveCellIndex {
    /// Validates `layout` and derives its canonical active-cell index.
    pub fn build(layout: &PbcPublicLayout) -> Result<Self, PbcError> {
        layout.validate()?;

        // Every coordinate has exactly `hash_count` real replicas, and every
        // bucket contributes exactly one dummy.  This reserves only active
        // cells instead of the potentially much larger padded rectangle.
        let active_cell_count = layout
            .parameters
            .universe_size
            .checked_mul(layout.parameters.hash_count)
            .and_then(|real| real.checked_add(layout.parameters.bucket_count))
            .ok_or(PbcError::SizeOverflow)?;
        let mut entries = Vec::with_capacity(active_cell_count);
        let mut bucket_offsets = Vec::with_capacity(layout.cells.len() + 1);
        bucket_offsets.push(0);

        for (bucket, row) in layout.cells.iter().enumerate() {
            for (slot, cell) in row.iter().enumerate() {
                if !matches!(cell, PbcCell::Padding) {
                    entries.push((bucket, slot, entries.len()));
                }
            }
            bucket_offsets.push(entries.len());
        }

        if entries.len() != active_cell_count {
            return Err(PbcError::InvalidLayout(
                "active-cell accounting does not equal real replicas plus dummies".into(),
            ));
        }

        Ok(Self { entries, bucket_offsets })
    }

    /// Iterates `(bucket, slot, flat_nonpadding_index)` in canonical order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.entries.iter().copied()
    }

    /// Returns the number of active (real plus dummy) cells in all buckets.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the index contains no real or dummy cells.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the active width/count of `bucket`, or `None` if it is absent.
    ///
    /// Since padding is the only inactive cell kind, this value is both the
    /// number of selectable cells and the width of the bucket's flattened
    /// active slice.
    pub fn bucket_active_width(&self, bucket: usize) -> Option<usize> {
        let end = bucket.checked_add(1)?;
        self.bucket_offsets.get(bucket..=end).map(|range| range[1] - range[0])
    }

    /// Alias emphasizing that the active width is a cell count.
    pub fn bucket_active_count(&self, bucket: usize) -> Option<usize> {
        self.bucket_active_width(bucket)
    }

    /// Returns the active entries for `bucket` in canonical slot order.
    pub fn bucket_iter(
        &self,
        bucket: usize,
    ) -> Option<impl Iterator<Item = (usize, usize, usize)> + '_> {
        let end = bucket.checked_add(1)?;
        let range = self.bucket_offsets.get(bucket..=end)?;
        Some(self.entries[range[0]..range[1]].iter().copied())
    }

    /// Returns all per-bucket active widths/counts in bucket order.
    pub fn bucket_active_widths(&self) -> impl Iterator<Item = usize> + '_ {
        self.bucket_offsets.windows(2).map(|range| range[1] - range[0])
    }
}

fn encode_nested_usizes(out: &mut Vec<u8>, values: &[Vec<usize>]) -> Result<(), PbcError> {
    out.extend(u64::try_from(values.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
    for row in values {
        out.extend(u64::try_from(row.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        for &value in row {
            out.extend(u64::try_from(value).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        }
    }
    Ok(())
}

fn encode_cells(out: &mut Vec<u8>, cells: &[Vec<PbcCell>]) -> Result<(), PbcError> {
    out.extend(u64::try_from(cells.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
    for row in cells {
        out.extend(u64::try_from(row.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes());
        for cell in row {
            match cell {
                PbcCell::Real { coordinate, replica } => {
                    out.push(0);
                    out.extend(
                        u64::try_from(*coordinate)
                            .map_err(|_| PbcError::SizeOverflow)?
                            .to_le_bytes(),
                    );
                    out.extend(
                        u64::try_from(*replica).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes(),
                    );
                }
                PbcCell::Dummy => out.push(1),
                PbcCell::Padding => out.push(2),
            }
        }
    }
    Ok(())
}

use std::time::Instant;

use schedule::{deterministic_matching, schedule_from_owners};

/// Stable names used by [`PbcDiagnosticReport::performance_values`].
pub const PERFORMANCE_CATEGORIES: [&str; 7] = [
    "pbc_layout_hashing",
    "pbc_matching",
    "pbc_real_selector_packages",
    "pbc_dummy_selector_packages",
    "pbc_bucket_cell_evaluations",
    "pbc_bucket_reductions",
    "pbc_padding_overhead",
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Measurements from one attempted PBC layout generation.
pub struct PbcDiagnosticSample {
    /// Number of seed attempts performed.
    pub attempts: u32,
    /// Accepted attempt, if any.
    pub accepted_attempt: Option<u32>,
    /// Accepted rectangular bucket width, if any.
    pub bucket_width: Option<usize>,
    /// Count of width-limit failures.
    pub bucket_width_failures: u32,
    /// Count of matching failures.
    pub no_perfect_schedule_failures: u32,
    /// Layout construction time in nanoseconds.
    pub layout_time_nanos: u128,
    /// Matching time in nanoseconds.
    pub matching_time_nanos: u128,
}

impl PbcDiagnosticSample {
    /// Returns whether this sample produced an accepted layout.
    pub fn accepted(&self) -> bool {
        self.accepted_attempt.is_some()
    }
}

/// Measures one complete v1 key-layout generation attempt sequence.
///
/// The support is supplied together with the root seed so this API models the
/// supported honest key-generation order: the sparse support is fixed before
/// layout randomness is derived, and failed layout seeds are retried without
/// changing the support.  Callers cannot provide a pre-built public layout or
/// invoke the exact scheduler directly.  This keeps the diagnostic from
/// turning a long-lived public layout into an API for scheduling arbitrary
/// later supports.
///
/// The returned sample contains only aggregate timing, retry, and rectangular
/// layout information.  It never contains the support, selected slots, or the
/// private schedule.  Structural errors are returned immediately.  Exhausting
/// the declared retry budget is represented by an accepted-less sample so a
/// caller can include the failure in an aggregate success-rate report.
pub fn measure_key_layout(
    parameters: &PbcParameters,
    root_seed: PbcRootSeed,
    support: &[usize],
) -> Result<PbcDiagnosticSample, PbcError> {
    let mut layout_time_nanos = 0_u128;
    let mut matching_time_nanos = 0_u128;
    let mut bucket_width_failures = 0_u32;
    let result = schedule::retry_layout(
        parameters,
        root_seed,
        support,
        |seed, attempt| {
            let start = Instant::now();
            let result = PbcPublicLayout::build(parameters, seed, attempt);
            layout_time_nanos = layout_time_nanos.saturating_add(start.elapsed().as_nanos());
            if matches!(&result, Err(PbcError::BucketWidthExceeded)) {
                bucket_width_failures += 1;
            }
            result
        },
        |layout, support| {
            let start = Instant::now();
            let owners = deterministic_matching(layout, support);
            matching_time_nanos = matching_time_nanos.saturating_add(start.elapsed().as_nanos());
            // Validation remains outside the matching timer.
            schedule_from_owners(layout, support, owners?)
        },
    );
    let (attempts, accepted_attempt, bucket_width, no_perfect_schedule_failures) = match result {
        Ok(generated) => {
            let layout = generated.public_layout;
            (
                layout.accepted_attempt + 1,
                Some(layout.accepted_attempt),
                Some(layout.bucket_width),
                layout.accepted_attempt - bucket_width_failures,
            )
        }
        Err(PbcError::SeedAttemptsExhausted(diagnostics)) => {
            (diagnostics.attempts, None, None, diagnostics.no_perfect_schedule_failures)
        }
        Err(error) => return Err(error),
    };
    Ok(PbcDiagnosticSample {
        attempts,
        accepted_attempt,
        bucket_width,
        bucket_width_failures,
        no_perfect_schedule_failures,
        layout_time_nanos,
        matching_time_nanos,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Aggregated layout-generation and cost measurements.
pub struct PbcDiagnosticReport {
    /// Names of reported performance categories.
    pub performance_categories: Vec<String>,
    /// Stable category-keyed values. Timing values are nanoseconds; count
    /// values use the natural units named by the corresponding field below.
    pub performance_values: std::collections::BTreeMap<String, u128>,
    /// Parameter set used by the samples.
    pub parameters: PbcParameters,
    /// Number of recorded trials.
    pub trials: u64,
    /// Number of accepted trials.
    pub accepted_trials: u64,
    /// Number of first-attempt successes.
    pub first_attempt_successes: u64,
    /// Cumulative successes indexed by attempt number.
    pub cumulative_success_by_attempt: Vec<u64>,
    /// Aggregate width-limit failures.
    pub bucket_width_failures: u64,
    /// Aggregate matching failures.
    pub no_perfect_schedule_failures: u64,
    /// Minimum accepted bucket width.
    pub bucket_width_min: Option<usize>,
    /// Median accepted bucket width.
    pub bucket_width_p50: Option<usize>,
    /// 95th-percentile accepted bucket width.
    pub bucket_width_p95: Option<usize>,
    /// Maximum accepted bucket width.
    pub bucket_width_max: Option<usize>,
    /// Number of real cells.
    pub real_cells: u128,
    /// Minimum padded storage size.
    pub padded_storage_min: Option<u128>,
    /// Median padded storage size.
    pub padded_storage_p50: Option<u128>,
    /// 95th-percentile padded storage size.
    pub padded_storage_p95: Option<u128>,
    /// Maximum padded storage size.
    pub padded_storage_max: Option<u128>,
    /// Median padded-to-real storage ratio.
    pub padded_to_real_ratio_p50: Option<f64>,
    /// Aggregate layout time in nanoseconds.
    pub layout_time_nanos: u128,
    /// Aggregate matching time in nanoseconds.
    pub matching_time_nanos: u128,
    /// Number of real selector packages.
    pub selector_real_packages: u128,
    /// Number of dummy selector packages.
    pub selector_dummy_packages: u128,
    /// Total selector packages.
    pub selector_total_packages: u128,
    /// Number of bucket-cell evaluations.
    pub bucket_cell_evaluations: u128,
    /// Number of bucket reductions.
    pub bucket_reductions: u128,
    /// Padding overhead in storage units.
    pub padding_overhead: u128,
}

#[derive(Clone, Debug)]
/// Accumulates opt-in PBC construction diagnostics without ciphertext data.
pub struct PbcDiagnosticAggregator {
    parameters: PbcParameters,
    samples: Vec<PbcDiagnosticSample>,
}

impl PbcDiagnosticAggregator {
    /// Starts an empty report for `parameters`.
    pub fn new(parameters: PbcParameters) -> Self {
        Self { parameters, samples: Vec::new() }
    }

    /// Adds one layout-generation sample to the aggregate.
    pub fn record(&mut self, sample: PbcDiagnosticSample) {
        self.samples.push(sample);
    }

    /// Borrows all recorded samples.
    pub fn samples(&self) -> &[PbcDiagnosticSample] {
        &self.samples
    }

    /// Computes the aggregate report, including percentile storage estimates.
    pub fn finish(&self) -> PbcDiagnosticReport {
        let accepted: Vec<&PbcDiagnosticSample> =
            self.samples.iter().filter(|sample| sample.accepted()).collect();
        let mut widths: Vec<usize> =
            accepted.iter().filter_map(|sample| sample.bucket_width).collect();
        widths.sort_unstable();
        let padded: Vec<u128> = widths
            .iter()
            .map(|&width| self.parameters.bucket_count as u128 * width as u128)
            .collect();
        let accepted_trials = accepted.len() as u64;
        let mut cumulative_success_by_attempt =
            vec![0_u64; self.parameters.max_seed_attempts as usize];
        for sample in &accepted {
            if let Some(attempt) = sample.accepted_attempt {
                for count in cumulative_success_by_attempt.iter_mut().skip(attempt as usize) {
                    *count += 1;
                }
            }
        }
        let real_cells_per_layout = (self.parameters.hash_count as u128)
            .saturating_mul(self.parameters.universe_size as u128);
        let real_cells = real_cells_per_layout.saturating_mul(accepted_trials as u128);
        let selector_dummy_packages =
            (self.parameters.bucket_count as u128).saturating_mul(accepted_trials as u128);
        let selector_real_packages = real_cells;
        let bucket_cell_evaluations: u128 = padded.iter().sum();
        let logical_nonpadding = selector_real_packages.saturating_add(selector_dummy_packages);
        let padding_overhead = bucket_cell_evaluations.saturating_sub(logical_nonpadding);
        let layout_time_nanos: u128 = accepted.iter().map(|sample| sample.layout_time_nanos).sum();
        let matching_time_nanos: u128 =
            accepted.iter().map(|sample| sample.matching_time_nanos).sum();
        let bucket_reductions = self.parameters.bucket_count as u128 * accepted_trials as u128;
        let selector_total_packages =
            selector_real_packages.saturating_add(selector_dummy_packages);
        let mut performance_values = std::collections::BTreeMap::new();
        performance_values.insert("pbc_layout_hashing".to_owned(), layout_time_nanos);
        performance_values.insert("pbc_matching".to_owned(), matching_time_nanos);
        performance_values.insert("pbc_real_selector_packages".to_owned(), selector_real_packages);
        performance_values
            .insert("pbc_dummy_selector_packages".to_owned(), selector_dummy_packages);
        performance_values
            .insert("pbc_bucket_cell_evaluations".to_owned(), bucket_cell_evaluations);
        performance_values.insert("pbc_bucket_reductions".to_owned(), bucket_reductions);
        performance_values.insert("pbc_padding_overhead".to_owned(), padding_overhead);
        PbcDiagnosticReport {
            performance_categories: PERFORMANCE_CATEGORIES
                .iter()
                .map(|category| (*category).to_owned())
                .collect(),
            performance_values,
            parameters: self.parameters.clone(),
            trials: self.samples.len() as u64,
            accepted_trials,
            first_attempt_successes: accepted
                .iter()
                .filter(|sample| sample.accepted_attempt == Some(0))
                .count() as u64,
            cumulative_success_by_attempt,
            bucket_width_failures: self
                .samples
                .iter()
                .map(|sample| sample.bucket_width_failures as u64)
                .sum(),
            no_perfect_schedule_failures: self
                .samples
                .iter()
                .map(|sample| sample.no_perfect_schedule_failures as u64)
                .sum(),
            bucket_width_min: widths.first().copied(),
            bucket_width_p50: percentile(&widths, 50),
            bucket_width_p95: percentile(&widths, 95),
            bucket_width_max: widths.last().copied(),
            real_cells,
            padded_storage_min: padded.first().copied(),
            padded_storage_p50: percentile(&padded, 50),
            padded_storage_p95: percentile(&padded, 95),
            padded_storage_max: padded.last().copied(),
            padded_to_real_ratio_p50: percentile(&padded, 50)
                .map(|value| value as f64 / real_cells_per_layout.max(1) as f64),
            layout_time_nanos,
            matching_time_nanos,
            selector_real_packages,
            selector_dummy_packages,
            selector_total_packages,
            bucket_cell_evaluations,
            bucket_reductions,
            padding_overhead,
        }
    }
}

fn percentile<T: Copy>(values: &[T], percentile: usize) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    // Nearest-rank percentile: p95 of two observations is the larger one.
    let index = ((values.len() * percentile).saturating_add(99) / 100)
        .saturating_sub(1)
        .min(values.len() - 1);
    Some(values[index])
}

impl PbcDiagnosticReport {
    /// Serializes the aggregate report as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Formats the aggregate report as human-readable text.
    pub fn to_text(&self) -> String {
        let optional =
            |value: Option<usize>| value.map_or_else(|| "n/a".to_owned(), |v| v.to_string());
        format!(
            concat!(
                "PBC diagnostic\n",
                "parameters: nu={} h={} c={} k={} max_seed_attempts={} profile={:?}\n",
                "trials: {} accepted={} first_attempt_successes={} cumulative_success_by_attempt={:?}\n",
                "failures: bucket_width={} no_perfect_schedule={}\n",
                "bucket_width: min={} p50={} p95={} max={} padded_storage: min={} p50={} p95={} max={} ratio_p50={:?}\n",
                "cells (accepted-trial totals): real={} selector_real={} selector_dummy={} selector_total={} bucket_cell_evaluations={} bucket_reductions={} padding_overhead={}\n",
                "time_nanos: layout={} matching={}\n",
                "performance_categories: {}\n",
                "performance_values: {:?}"
            ),
            self.parameters.universe_size,
            self.parameters.support_weight,
            self.parameters.hash_count,
            self.parameters.bucket_count,
            self.parameters.max_seed_attempts,
            self.parameters.profile,
            self.trials,
            self.accepted_trials,
            self.first_attempt_successes,
            self.cumulative_success_by_attempt,
            self.bucket_width_failures,
            self.no_perfect_schedule_failures,
            optional(self.bucket_width_min),
            optional(self.bucket_width_p50),
            optional(self.bucket_width_p95),
            optional(self.bucket_width_max),
            self.padded_storage_min.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_storage_p50.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_storage_p95.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_storage_max.map_or_else(|| "n/a".to_owned(), |v| v.to_string()),
            self.padded_to_real_ratio_p50,
            self.real_cells,
            self.selector_real_packages,
            self.selector_dummy_packages,
            self.selector_total_packages,
            self.bucket_cell_evaluations,
            self.bucket_reductions,
            self.padding_overhead,
            self.layout_time_nanos,
            self.matching_time_nanos,
            PERFORMANCE_CATEGORIES.join(","),
            self.performance_values,
        )
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    fn sample(attempt: Option<u32>, width: Option<usize>) -> PbcDiagnosticSample {
        PbcDiagnosticSample {
            attempts: 3,
            accepted_attempt: attempt,
            bucket_width: width,
            bucket_width_failures: 1,
            no_perfect_schedule_failures: 1,
            layout_time_nanos: 7,
            matching_time_nanos: 11,
        }
    }

    #[test]
    fn aggregation_reports_percentiles_and_overhead() {
        let parameters = PbcParameters::custom(10, 2, 3, 4, 3, None);
        let mut aggregate = PbcDiagnosticAggregator::new(parameters);
        aggregate.record(sample(Some(0), Some(4)));
        aggregate.record(sample(Some(2), Some(6)));
        aggregate.record(sample(None, None));
        let report = aggregate.finish();
        assert_eq!(report.trials, 3);
        assert_eq!(report.accepted_trials, 2);
        assert_eq!(report.first_attempt_successes, 1);
        assert_eq!(report.cumulative_success_by_attempt, vec![1, 1, 2]);
        assert_eq!(report.bucket_width_min, Some(4));
        assert_eq!(report.bucket_width_p50, Some(4));
        assert_eq!(report.bucket_width_p95, Some(6));
        assert_eq!(report.bucket_width_max, Some(6));
        assert_eq!(report.real_cells, report.accepted_trials as u128 * 3 * 10);
        assert_eq!(report.padded_storage_min, Some(16));
        assert_eq!(report.padded_storage_max, Some(24));
        assert_eq!(report.bucket_cell_evaluations, 40);
        assert_eq!(report.bucket_reductions, 8);
        assert_eq!(report.padding_overhead, 0);
        assert_eq!(report.bucket_width_failures, 3);
        assert_eq!(report.no_perfect_schedule_failures, 3);
        assert_eq!(report.performance_values["pbc_layout_hashing"], 14);
        assert_eq!(report.performance_values["pbc_matching"], 22);
        assert_eq!(report.performance_values["pbc_real_selector_packages"], 60);
        assert_eq!(report.performance_values["pbc_dummy_selector_packages"], 8);
    }

    #[test]
    fn text_and_json_use_stable_category_names() {
        let report =
            PbcDiagnosticAggregator::new(PbcParameters::custom(4, 1, 2, 2, 1, None)).finish();
        let text = report.to_text();
        for category in PERFORMANCE_CATEGORIES {
            assert!(text.contains(category));
            assert!(report.to_json().unwrap().contains(category));
        }
        assert!(report.to_json().unwrap().contains("cumulative_success_by_attempt"));
    }

    #[test]
    fn key_layout_measurement_is_deterministic_and_keeps_support_private() {
        let parameters = PbcParameters::custom(10, 2, 3, 4, 3, None);
        let first = measure_key_layout(&parameters, PbcRootSeed([23; 32]), &[1, 7]).unwrap();
        let second = measure_key_layout(&parameters, PbcRootSeed([23; 32]), &[1, 7]).unwrap();
        assert_eq!(first.attempts, second.attempts);
        assert_eq!(first.accepted_attempt, second.accepted_attempt);
        assert_eq!(first.bucket_width, second.bucket_width);
        assert_eq!(first.bucket_width_failures, second.bucket_width_failures);
        assert_eq!(first.no_perfect_schedule_failures, second.no_perfect_schedule_failures);
        assert!(format!("{first:?}").contains("accepted_attempt"));
        assert!(!format!("{first:?}").contains("1, 7"));
    }

    #[test]
    fn key_layout_measurement_rejects_invalid_support_before_reporting() {
        let parameters = PbcParameters::custom(10, 2, 3, 4, 3, None);
        assert!(matches!(
            measure_key_layout(&parameters, PbcRootSeed([24; 32]), &[1]),
            Err(PbcError::SupportSize)
        ));
        assert!(matches!(
            measure_key_layout(&parameters, PbcRootSeed([24; 32]), &[1, 1]),
            Err(PbcError::InvalidSupport)
        ));
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A public LWR vector arranged in PBC bucket/slot order.
pub struct PbcEncodedPublicVector {
    /// Layout identity for the `values` matrix.
    pub layout_id: PbcLayoutId,
    /// Modulus applied to every routed coordinate.
    pub modulus: usize,
    /// Rectangular bucket rows in public layout order.
    pub values: Vec<Vec<usize>>,
}

impl PbcEncodedPublicVector {
    /// Stable content identity for the encryptor-supplied public vector.
    ///
    /// The layout and modulus are part of the identity, so two encryptions
    /// under one setup may use different `D0` values without colliding, while
    /// setup artifacts (which do not contain this value) remain reusable.
    pub fn identity(&self) -> [u8; 32] {
        artifacts::public_vector_id(self)
    }

    /// Routes a `u64` vector through real cells and zeroes dummy/padding cells.
    pub fn route(
        layout: &PbcPublicLayout,
        public_vector: &[u64],
        modulus: usize,
    ) -> Result<Self, PbcError> {
        let modulus_u64 = u64::try_from(modulus).map_err(|_| PbcError::SizeOverflow)?;
        Self::route_by_coordinate(layout, public_vector.len(), modulus, |coordinate| {
            usize::try_from(public_vector[coordinate] % modulus_u64)
                .map_err(|_| PbcError::SizeOverflow)
        })
    }

    /// Derives and routes a deterministic public vector from a label.
    pub fn from_label(
        layout: &PbcPublicLayout,
        label: &[u8],
        modulus: usize,
    ) -> Result<Self, PbcError> {
        let vector =
            derive_lwr_vector(layout.layout_id, label, layout.parameters.universe_size, modulus)?;
        Self::route_usize(layout, &vector, modulus)
    }

    /// Routes a vector already represented as `usize` residues.
    pub fn route_usize(
        layout: &PbcPublicLayout,
        public_vector: &[usize],
        modulus: usize,
    ) -> Result<Self, PbcError> {
        Self::route_by_coordinate(layout, public_vector.len(), modulus, |coordinate| {
            Ok(public_vector[coordinate] % modulus)
        })
    }

    /// Checks layout identity, dimensions, modulus bounds, and cell semantics.
    pub fn validate(&self, layout: &PbcPublicLayout) -> Result<(), PbcError> {
        layout.validate()?;
        if self.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        self.validate_values(layout)
    }

    /// Routes one coordinate lookup at a time after validating the layout.
    ///
    /// Keeping the lookup callback local to the real-cell branch means
    /// `route_usize` does not allocate an intermediate `u64` vector.  The
    /// constructor performs one layout validation and one output-shape check;
    /// the public [`Self::validate`] method remains the fail-closed entry point
    /// for callers handling deserialized or otherwise untrusted values.
    fn route_by_coordinate<F>(
        layout: &PbcPublicLayout,
        vector_len: usize,
        modulus: usize,
        mut value_at: F,
    ) -> Result<Self, PbcError>
    where
        F: FnMut(usize) -> Result<usize, PbcError>,
    {
        layout.validate()?;
        if modulus == 0 {
            return Err(PbcError::InvalidParameters("LWR modulus must be positive".into()));
        }
        if vector_len != layout.parameters.universe_size {
            return Err(PbcError::InvalidLayout("public vector dimension mismatch".into()));
        }
        let values = layout
            .cells
            .iter()
            .map(|row| {
                // Preserve rectangular storage, but make inactive cells zero
                // so only real coordinates can contribute to a bucket sum.
                row.iter()
                    .map(|cell| match cell {
                        PbcCell::Real { coordinate, .. } => value_at(*coordinate),
                        PbcCell::Dummy | PbcCell::Padding => Ok(0),
                    })
                    .collect::<Result<Vec<_>, PbcError>>()
            })
            .collect::<Result<Vec<_>, PbcError>>()?;
        let encoded = Self { layout_id: layout.layout_id, modulus, values };
        encoded.validate_values(layout)?;
        Ok(encoded)
    }

    /// Validates output structure against a layout that has already been
    /// validated by the caller.
    fn validate_values(&self, layout: &PbcPublicLayout) -> Result<(), PbcError> {
        if self.modulus == 0 || self.values.len() != layout.parameters.bucket_count {
            return Err(PbcError::InvalidLayout("encoded vector shape or modulus mismatch".into()));
        }
        for (bucket, row) in self.values.iter().enumerate() {
            if row.len() != layout.bucket_width {
                return Err(PbcError::InvalidLayout("encoded vector is not rectangular".into()));
            }
            for (slot, &value) in row.iter().enumerate() {
                if value >= self.modulus {
                    return Err(PbcError::InvalidLayout(
                        "encoded value is outside the modulus".into(),
                    ));
                }
                if matches!(layout.cells[bucket][slot], PbcCell::Dummy | PbcCell::Padding) &&
                    value != 0
                {
                    return Err(PbcError::InvalidLayout("dummy or padding value is nonzero".into()));
                }
            }
        }
        Ok(())
    }
}

/// Serializable binding between a routed public vector and one DSL family.
///
/// The host binding stores one public residue and its active row-major order
/// for every real or dummy cell, as defined by
/// [`crate::noise_refresh::pbc::PbcActiveCellIndex`]. Padding has no family element. At the
/// DSL runtime boundary, each residue `a` is materialized as the public
/// monomial factor `X^a` consumed by the native selector binding; the declared `(1, 1)` describes
/// the matrix shape only, not a scalar-output encoding. This is a declaration
/// and runtime-binding boundary: [`Self::input_family`] and
/// [`Self::artifact_input_family`] creates one structural family input, while
/// [`Self::values_u64`] supplies its host residue values. They do not pack one
/// DSL node per cell.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PbcPublicVectorFamilyBinding {
    /// Public layout to which the family and values are bound.
    pub layout_id: PbcLayoutId,
    /// Modulus used when the vector was routed.
    pub modulus: usize,
    /// Number of public-factor family elements, excluding padding cells.
    pub family_count: usize,
    /// Flattened active values in canonical row-major order.
    values: Vec<u64>,
}

impl PbcPublicVectorFamilyBinding {
    /// Stable identity of the routed family payload and its public schema.
    /// This identity is intentionally distinct from the setup/layout ID and
    /// must be included in encrypt-stage artifact descriptors.
    pub fn identity(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"mxx-exponent-lut/pbc/public-family-binding/v1");
        digest.update(self.layout_id.0);
        digest.update((self.modulus as u64).to_le_bytes());
        digest.update((self.family_count as u64).to_le_bytes());
        for value in &self.values {
            digest.update(value.to_le_bytes());
        }
        digest.finalize().into()
    }

    /// Creates a binding from an already routed public vector.
    ///
    /// The resulting flat order is the active-cell order shared by PBC
    /// selector families: real cells and the per-bucket dummy are retained,
    /// while padding cells are omitted.
    pub fn from_encoded(
        layout: &PbcPublicLayout,
        encoded: &PbcEncodedPublicVector,
    ) -> Result<Self, PbcError> {
        layout.validate()?;
        if encoded.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        encoded.validate_values(layout)?;

        let mut values = Vec::new();
        for (bucket, row) in encoded.values.iter().enumerate() {
            for (slot, &value) in row.iter().enumerate() {
                if !matches!(layout.cells[bucket][slot], PbcCell::Padding) {
                    values.push(u64::try_from(value).map_err(|_| PbcError::SizeOverflow)?);
                }
            }
        }
        let binding = Self {
            layout_id: layout.layout_id,
            modulus: encoded.modulus,
            family_count: values.len(),
            values,
        };
        binding.validate(layout)?;
        Ok(binding)
    }

    /// Creates an encryptor-supplied D0 family with an explicit transform
    /// column count. D0 is not the routed PBC cell family: its cardinality is
    /// `r = v * log2(Q_exp)` and must match the frozen PRF profile.
    pub fn from_d0_values(
        layout: &PbcPublicLayout,
        values: Vec<u64>,
        modulus: usize,
        expected_count: usize,
    ) -> Result<Self, PbcError> {
        let binding =
            Self { layout_id: layout.layout_id, modulus, family_count: values.len(), values };
        binding.validate_d0(layout, expected_count)?;
        Ok(binding)
    }

    /// Validates layout identity, modulus, count, and value bounds.
    ///
    /// Call this after importing serialized binding metadata and before using
    /// either the DSL family declaration or runtime values.
    pub fn validate(&self, layout: &PbcPublicLayout) -> Result<(), PbcError> {
        layout.validate()?;
        if self.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        let expected_count = layout
            .parameters
            .universe_size
            .checked_mul(layout.parameters.hash_count)
            .and_then(|real| real.checked_add(layout.parameters.bucket_count))
            .ok_or(PbcError::SizeOverflow)?;
        self.validate_count_and_values(expected_count)?;
        Ok(())
    }

    /// Validates a D0 family against the exact transform column count.
    pub fn validate_d0(
        &self,
        layout: &PbcPublicLayout,
        expected_count: usize,
    ) -> Result<(), PbcError> {
        layout.validate()?;
        if self.layout_id != layout.layout_id {
            return Err(PbcError::LayoutIdentityMismatch);
        }
        self.validate_count_and_values(expected_count)
    }

    fn validate_count_and_values(&self, expected_count: usize) -> Result<(), PbcError> {
        if self.modulus == 0 ||
            self.family_count != expected_count ||
            self.values.len() != self.family_count
        {
            return Err(PbcError::InvalidLayout(
                "public vector family binding shape mismatch".into(),
            ));
        }
        let modulus = u64::try_from(self.modulus).map_err(|_| PbcError::SizeOverflow)?;
        if self.values.iter().any(|&value| value >= modulus) {
            return Err(PbcError::InvalidLayout(
                "public vector family value is outside the modulus".into(),
            ));
        }
        Ok(())
    }

    /// Returns the canonical serialized public values as `u64` residues.
    pub fn values_u64(&self) -> &[u64] {
        &self.values
    }

    /// Converts the canonical public values to `usize` residues.
    pub fn values_usize(&self) -> Result<Vec<usize>, PbcError> {
        self.values
            .iter()
            .copied()
            .map(|value| usize::try_from(value).map_err(|_| PbcError::SizeOverflow))
            .collect()
    }

    /// Declares one structural `(1, 1)` public-factor input family.
    ///
    /// The family has one element per active cell and is intended to be read
    /// through dynamic indexing inside the reusable bucket loop. Runtime
    /// binding materializes each host residue `a` as the monomial `X^a`;
    /// `(1, 1)` is the matrix shape, not a scalar-output representation.
    pub fn input_family(&self, ring: &Ring, name: impl Into<String>) -> Family<Mat> {
        ring.input_family(name, self.family_count, (1, 1))
    }

    /// Declares one artifact-backed `(1, 1)` public-factor family with the
    /// same active count.
    ///
    /// The artifact family contains only public routed values; it carries no
    /// support coordinate or private schedule information.
    pub fn artifact_input_family(
        &self,
        ring: &Ring,
        production_id: ProductionId,
        artifact_name: impl Into<String>,
    ) -> Family<Mat> {
        ring.family_artifact_input(production_id, artifact_name, self.family_count, (1, 1))
    }
}

/// Derives the public LWR vector associated with a label and domain tag.
///
/// Each coordinate is hashed with the layout identity, modulus, label, and
/// coordinate index. Rejection sampling removes modulo bias before reducing
/// into `[0, modulus)`, so this function produces public residues without
/// consulting a private support or schedule.
pub fn derive_lwr_vector(
    layout_id: PbcLayoutId,
    label: &[u8],
    universe_size: usize,
    modulus: usize,
) -> Result<Vec<usize>, PbcError> {
    if modulus == 0 {
        return Err(PbcError::InvalidParameters("LWR modulus must be positive".into()));
    }
    let modulus_u64 = u64::try_from(modulus).map_err(|_| PbcError::SizeOverflow)?;
    let range = 1u128 << 64;
    let limit = (range / u128::from(modulus_u64)) * u128::from(modulus_u64);
    let mut values = Vec::with_capacity(universe_size);
    for coordinate in 0..universe_size {
        let coordinate = u64::try_from(coordinate).map_err(|_| PbcError::SizeOverflow)?;
        let mut nonce = 0u64;
        loop {
            let mut hasher = Sha256::new();
            hasher.update(b"mxx-exponent-lut/sparse-lwr/vector/v1");
            hasher.update(layout_id.0);
            hasher.update(modulus_u64.to_le_bytes());
            hasher.update(
                u64::try_from(label.len()).map_err(|_| PbcError::SizeOverflow)?.to_le_bytes(),
            );
            hasher.update(label);
            hasher.update(coordinate.to_le_bytes());
            hasher.update(nonce.to_le_bytes());
            let digest = hasher.finalize();
            // Rejection sampling turns the first 64 digest bits into an
            // unbiased residue; reduction is reached only after the largest
            // divisible prefix of the 64-bit domain is accepted.
            let value =
                u64::from_le_bytes(digest[..8].try_into().map_err(|_| PbcError::SizeOverflow)?);
            if u128::from(value) < limit {
                values.push((value % modulus_u64) as usize);
                break;
            }
            nonce = nonce.checked_add(1).ok_or(PbcError::HashNonceOverflow)?;
        }
    }
    Ok(values)
}

#[cfg(test)]
pub(crate) mod routing_tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::noise_refresh::pbc::{PbcActiveCellIndex, PbcLayoutSeed, PbcParameters};

    pub(crate) fn layout_with_padding() -> PbcPublicLayout {
        let parameters = PbcParameters::custom(4, 2, 2, 3, 1, None);
        for byte in 0..=u8::MAX {
            let layout = PbcPublicLayout::build(&parameters, PbcLayoutSeed([byte; 32]), 0)
                .expect("toy layout construction");
            if layout.cells.iter().flatten().any(|cell| matches!(cell, PbcCell::Padding)) {
                return layout;
            }
        }
        panic!("toy layout search did not produce padding");
    }

    #[test]
    fn family_binding_preserves_active_order_and_excludes_padding() {
        let layout = layout_with_padding();
        let vector = PbcEncodedPublicVector::route_usize(&layout, &[11, 22, 33, 44], 17)
            .expect("route toy vector");
        let binding =
            PbcPublicVectorFamilyBinding::from_encoded(&layout, &vector).expect("bind toy vector");
        let active = PbcActiveCellIndex::build(&layout).expect("index valid layout");

        let expected_locations = layout
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
            expected_locations
        );
        assert!(layout.cells.iter().flatten().any(|cell| matches!(cell, PbcCell::Padding)));

        let expected_values = expected_locations
            .iter()
            .map(|&(bucket, slot)| vector.values[bucket][slot] as u64)
            .collect::<Vec<_>>();
        assert_eq!(binding.values_u64(), expected_values.as_slice());
        assert_eq!(binding.family_count, active.len());
        assert_eq!(
            binding.values_usize().unwrap(),
            expected_values.iter().map(|&v| v as usize).collect::<Vec<_>>()
        );
    }

    #[test]
    fn family_binding_declares_a_scalar_family_with_the_active_count() {
        let layout = layout_with_padding();
        let vector = PbcEncodedPublicVector::route_usize(&layout, &[11, 22, 33, 44], 17)
            .expect("route toy vector");
        let binding =
            PbcPublicVectorFamilyBinding::from_encoded(&layout, &vector).expect("bind toy vector");
        let ring = Ring::new(257, 8);
        let family: Family<Mat> = binding.input_family(&ring, "pbc-values");
        let active_count = PbcActiveCellIndex::build(&layout).unwrap().len();
        assert_eq!(binding.family_count, active_count);
        assert_eq!(family.count(), &mxx_ir_core::IntExpr::constant(active_count));
    }

    #[test]
    fn neutral_scalar_and_family_artifact_inputs_remain_lazy_and_typed() {
        use mxx_dsl::DslContext;
        use mxx_ir_core::artifact::{Manifest, ManifestArtifact, SpecHash};

        let layout = layout_with_padding();
        let vector = PbcEncodedPublicVector::route_usize(&layout, &[11, 22, 33, 44], 17)
            .expect("route toy vector");
        let binding =
            PbcPublicVectorFamilyBinding::from_encoded(&layout, &vector).expect("bind toy vector");
        let ring = Ring::new(257, 8);
        let production = ProductionId { spec_hash: SpecHash([31; 32]), execution_nonce: [32; 32] };
        let scalar = ring.artifact_input(production.clone(), "pbc-scalar", (1, 1));
        let family = binding.artifact_input_family(&ring, production.clone(), "pbc-family");
        let graph = DslContext::new("pbc-neutral-artifact-inputs")
            .artifact_output("scalar", scalar)
            .unwrap()
            .artifact_output("family", family)
            .unwrap()
            .build()
            .unwrap();
        let matrix_type = artifacts::concrete_matrix_type(&ring, (1, 1)).unwrap();
        let manifest = Manifest {
            ir_version: mxx_ir_core::encoding::IR_VERSION,
            production_id: production.clone(),
            artifacts: BTreeMap::from([
                (
                    "pbc-scalar".into(),
                    ManifestArtifact {
                        artifact_type: ArtifactType::Matrix(matrix_type.clone()),
                        family_count: None,
                        layout: None,
                    },
                ),
                (
                    "pbc-family".into(),
                    ManifestArtifact {
                        artifact_type: ArtifactType::Matrix(matrix_type),
                        family_count: Some(binding.family_count),
                        layout: None,
                    },
                ),
            ]),
        };
        graph
            .validate_with_manifests(
                &mxx_ir_core::ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .expect("neutral artifact inputs validate against typed manifest");
    }

    #[test]
    fn family_binding_round_trip_and_layout_mismatch_fail_closed() {
        let parameters = PbcParameters::custom(4, 2, 2, 3, 1, None);
        let layout = layout_with_padding();
        let vector = PbcEncodedPublicVector::route_usize(&layout, &[11, 22, 33, 44], 17)
            .expect("route toy vector");
        let binding =
            PbcPublicVectorFamilyBinding::from_encoded(&layout, &vector).expect("bind toy vector");
        let encoded = serde_json::to_vec(&binding).expect("serialize binding");
        let decoded: PbcPublicVectorFamilyBinding =
            serde_json::from_slice(&encoded).expect("deserialize binding");
        decoded.validate(&layout).expect("round-trip binding validates");

        let other = PbcPublicLayout::build(&parameters, PbcLayoutSeed([77; 32]), 0)
            .expect("other toy layout");
        assert!(matches!(decoded.validate(&other), Err(PbcError::LayoutIdentityMismatch)));
    }
}

/// Public layout families supplied by the key/label compiler. These are typed
/// family inputs rather than host-packed values. `package_indices` is a public
/// 1x1 matrix family: its constant coefficient maps rectangular cells to the
/// non-padding selector family. Keeping the index in a matrix is intentional;
/// the artifact model has no first-class integer family artifacts. The other
/// two families provide the public rotation and active-cell mask.
pub struct PbcLayoutFamilies {
    layout_id: crate::noise_refresh::pbc::PbcLayoutId,
    encoded_vector_id: [u8; 32],
    rectangular_count: usize,
    package_indices: Family<Mat>,
    active_masks: Family<Mat>,
    shifts: Family<Mat>,
}

impl PbcLayoutFamilies {
    /// Construct the only layout-family handles accepted by the PBC compiler.
    /// Shape and identity are derived from the validated layout/vector; callers
    /// cannot inject arbitrary index, mask, or shift families.
    pub(crate) fn from_layout(
        ring: &Ring,
        layout: &PbcPublicLayout,
        encoded: &PbcEncodedPublicVector,
    ) -> Result<Self, PbcError> {
        layout.validate().map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        encoded.validate(layout).map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        let rectangular_count = layout.parameters.bucket_count * layout.bucket_width;
        let package_count = crate::noise_refresh::pbc::PbcActiveCellIndex::build(layout)
            .map_err(|_| PbcError::ArtifactIdentityMismatch)?
            .len();
        Ok(Self {
            layout_id: layout.layout_id,
            encoded_vector_id: artifacts::public_vector_id(encoded),
            rectangular_count,
            package_indices: ring.input_family(
                artifacts::canonical_family_name("package-indices", layout.layout_id, None),
                rectangular_count,
                (1, 1),
            ),
            active_masks: ring.input_family(
                artifacts::canonical_family_name("active-masks", layout.layout_id, None),
                rectangular_count,
                (1, 1),
            ),
            shifts: ring.input_family(
                artifacts::canonical_family_name("shifts", layout.layout_id, None),
                package_count,
                (1, 1),
            ),
        })
    }

    /// Construct all public layout families from a validated artifact
    /// manifest. Every family is artifact-backed, including package indices;
    /// the latter uses a 1x1 matrix whose constant coefficient is extracted
    /// inside the structural loop before dynamic selector-family indexing.
    pub fn from_layout_artifacts(
        ring: &Ring,
        layout: &PbcPublicLayout,
        encoded: &PbcEncodedPublicVector,
        production_id: ProductionId,
        manifest: &Manifest,
    ) -> Result<Self, PbcError> {
        let mut families = Self::from_layout(ring, layout, encoded)?;
        artifacts::validate_manifest_identity(manifest, &production_id)?;
        let rectangular_count = families.rectangular_count;
        let package_count = crate::noise_refresh::pbc::PbcActiveCellIndex::build(layout)
            .map_err(|_| PbcError::ArtifactIdentityMismatch)?
            .len();
        let mask_name = artifacts::public_family_artifact_name(encoded, "active-masks");
        let shift_name = artifacts::public_family_artifact_name(encoded, "shifts");
        let package_index_name = artifacts::public_family_artifact_name(encoded, "package-indices");
        let scalar_type = ArtifactType::Matrix(artifacts::concrete_matrix_type(ring, (1, 1))?);
        artifacts::require_family_descriptor(
            manifest,
            &package_index_name,
            scalar_type.clone(),
            rectangular_count,
            None,
        )
        .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        artifacts::require_family_descriptor(
            manifest,
            &mask_name,
            scalar_type.clone(),
            rectangular_count,
            None,
        )
        .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        artifacts::require_family_descriptor(
            manifest,
            &shift_name,
            ArtifactType::Matrix(artifacts::concrete_matrix_type(ring, (1, 1))?),
            package_count,
            None,
        )
        .map_err(|_| PbcError::ArtifactIdentityMismatch)?;
        families.package_indices = ring.family_artifact_input(
            production_id.clone(),
            package_index_name,
            rectangular_count,
            (1, 1),
        );
        families.active_masks =
            ring.family_artifact_input(production_id.clone(), mask_name, rectangular_count, (1, 1));
        families.shifts =
            ring.family_artifact_input(production_id, shift_name, package_count, (1, 1));
        Ok(families)
    }

    /// Public family carrying the canonical non-padding package indices.
    pub fn package_indices(&self) -> &Family<Mat> {
        &self.package_indices
    }
    /// Public family carrying the rectangular active-cell mask.
    pub fn active_masks(&self) -> &Family<Mat> {
        &self.active_masks
    }
    /// Public family carrying the canonical package shifts.
    pub fn shifts(&self) -> &Family<Mat> {
        &self.shifts
    }

    /// Returns the layout identity committed by these family handles.
    pub fn layout_id(&self) -> crate::noise_refresh::pbc::PbcLayoutId {
        self.layout_id
    }

    /// Returns the identity of the routed public vector.
    pub fn encoded_vector_id(&self) -> [u8; 32] {
        self.encoded_vector_id
    }
}
