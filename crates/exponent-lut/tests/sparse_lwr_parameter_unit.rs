#[allow(dead_code)]
mod test_utils;

use serde_json::Value;
use std::collections::BTreeMap;
use test_utils::sparse_lwr_parameter_support::*;

#[test]
fn reviewed_primary_tuple_derives_reviewed_values() {
    for (tuple, expected_minimum, expected_floor) in reviewed_phase1_tuple_grid()
        .into_iter()
        .zip([100.5636531151896, 100.56464330713312, 103.18615790798486])
        .zip([100_u64, 100, 103])
        .map(|((tuple, minimum), floor)| (tuple, minimum, floor))
    {
        assert_eq!(tuple.estimator_minimum_classical_bits, Some(expected_minimum));
        assert_eq!(tuple.estimator_security_bits, Some(expected_floor));
        assert_eq!(pinned_estimator_security_bits(&tuple).unwrap(), Some(expected_floor));
        let candidate = tuple.candidate().unwrap();
        let derived = candidate.derived().unwrap();
        assert_eq!(candidate.nu, tuple.nu);
        assert_eq!(candidate.h, tuple.h);
        assert_eq!(derived.p_prime, tuple.p / 2);
        assert_eq!(derived.delta, tuple.q_l / tuple.p);
        assert_eq!((derived.error_lower, derived.error_upper), (-64, 63));
    }
    assert_eq!(
        native_exp_layout(2048, 1usize << 16).unwrap(),
        mxx_exponent_lut::ExpLayout { stride_log2: 6, width_log2: 10 }
    );
    assert_eq!(
        native_exp_layout(4096, 1usize << 16).unwrap(),
        mxx_exponent_lut::ExpLayout { stride_log2: 5, width_log2: 11 }
    );
}

#[test]
fn estimator_evidence_is_optional_but_must_be_complete_when_present() {
    let mut tuple = reviewed_phase1_tuple_grid().remove(0);
    tuple.estimator_minimum_classical_bits = None;
    tuple.estimator_security_bits = None;
    assert_eq!(pinned_estimator_security_bits(&tuple).unwrap(), None);

    tuple.estimator_minimum_classical_bits = Some(100.5);
    assert!(pinned_estimator_security_bits(&tuple).unwrap_err().contains("incomplete"));

    tuple.estimator_security_bits = Some(99);
    assert!(pinned_estimator_security_bits(&tuple).unwrap_err().contains("invalid"));
}

#[test]
fn non_dividing_output_modulus_is_rejected() {
    assert!(SparseLwrCandidate::new("bad", 32, 4, 8, 31, vec![16]).is_err());
}

#[test]
fn incompatible_encoding_ring_is_rejected() {
    assert!(SparseLwrCandidate::new("bad", 32, 4, 8, 2, vec![10]).is_err());
}

#[test]
fn phase_one_grid_is_ordered_and_accepts_native_nonbinary_parameters() {
    let grid = reviewed_phase1_tuple_grid();
    assert_eq!(
        grid.iter().map(|tuple| (tuple.q_l, tuple.p, tuple.nu, tuple.h)).collect::<Vec<_>>(),
        vec![(2048, 16, 16384, 17), (4096, 32, 16384, 17), (4096, 32, 8192, 18)]
    );
    assert!(SparseLwrCandidate::new("native-p16", 16384, 17, 2048, 16, vec![65536]).is_ok());
    assert!(SparseLwrCandidate::new("native-p32", 8192, 18, 4096, 32, vec![65536]).is_ok());
    assert!(SparseLwrCandidate::new("bad", 896, 24, 8, 4, vec![10]).is_err());
    assert!(SparseLwrCandidate::new("bad", 896, 24, 9, 2, vec![32]).is_err());
}

#[test]
fn native_candidate_derives_exact_n_2_16_layout_independently_of_nu() {
    let candidate = SparseLwrCandidate::new(
        "native-q4096-p64",
        16384,
        17,
        4096,
        64,
        vec![8192, 16384, 32768, 65536],
    )
    .unwrap();
    let derived = candidate.derived().unwrap();
    assert_eq!(derived.p_prime, 32);
    assert_eq!(derived.delta, 64);
    assert_eq!((derived.error_lower, derived.error_upper), (-32, 31));
    let layout = native_exp_layout(4096, 1usize << 16).unwrap();
    assert_eq!(layout.stride_log2, 5);
    assert_eq!(layout.width_log2, 11);
    assert_eq!(layout.effective_modulus(1usize << 16), 4096);
    assert_eq!(candidate.n_enc_candidates.last().copied(), Some(1usize << 16));

    let changed_universe = SparseLwrCandidate::new(
        "native-q4096-p64-small-universe",
        32,
        8,
        4096,
        64,
        vec![1usize << 16],
    )
    .unwrap();
    assert_eq!(native_exp_layout(4096, 1usize << 16).unwrap(), layout);
    assert_eq!(changed_universe.derived().unwrap().delta, derived.delta);
}

#[test]
fn infinity_is_not_treated_as_security() {
    let mut attacks = BTreeMap::new();
    for attack in EXPECTED_ATTACKS {
        attacks.insert((*attack).to_owned(), serde_json::json!({"rop_log2": 200.0}));
    }
    attacks.get_mut("bkw").unwrap()["rop_log2"] = Value::Null;
    let report = EstimatorReport {
        schema_version: 1,
        repository_url: LATTICE_ESTIMATOR_REPOSITORY.to_owned(),
        git_commit: "e35f45b7976a90a79c3c6625a45bbc344c1abc67".to_owned(),
        sage_version: "SageMath version 10.7".to_owned(),
        python_version: "3.12.12".to_owned(),
        cost_model: "MATZOV".to_owned(),
        shape_model: "gsa".to_owned(),
        quantum: false,
        sample_count: "infinity".to_owned(),
        attacks,
        failures: Vec::new(),
        infinite_attacks: vec!["bkw".to_owned()],
        minimum_classical_bits: 200.0,
    };
    assert!(report.validate().is_ok());
    let mut numeric_infinite = report.clone();
    numeric_infinite.attacks.get_mut("bkw").unwrap()["rop_log2"] = serde_json::json!(200.0);
    assert!(numeric_infinite.validate().is_err());
    let mut unknown = report.clone();
    unknown.infinite_attacks.push("unknown".to_owned());
    assert!(unknown.validate().is_err());
    let mut duplicate = report;
    duplicate.infinite_attacks.push("bkw".to_owned());
    assert!(duplicate.validate().is_err());
}
