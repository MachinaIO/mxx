
    #[test]
    fn reviewed_primary_tuple_derives_reviewed_values() {
        let tuple = reviewed_phase1_tuple_grid().remove(1);
        let candidate = tuple.candidate().unwrap();
        let derived = candidate.derived().unwrap();
        assert_eq!(candidate.p, 2);
        assert_eq!(candidate.q, 16);
        assert_eq!(derived.h_prime, 34);
        assert_eq!(derived.delta, 8);
        assert_eq!((derived.error_lower, derived.error_upper), (-4, 3));
        assert_eq!(derived.w_mod, 32);
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
    fn phase_one_grid_is_ordered_and_rejects_non_phase1_parameters() {
        let grid = reviewed_phase1_tuple_grid();
        assert_eq!(
            grid.iter().map(|tuple| (tuple.q_l, tuple.p, tuple.nu, tuple.h)).collect::<Vec<_>>(),
            vec![(16, 2, 450, 31), (16, 2, 451, 31)]
        );
        assert!(SparseLwrCandidate::new("bad", 896, 24, 8, 4, vec![16]).is_err());
        assert!(SparseLwrCandidate::new("bad", 896, 24, 9, 2, vec![32]).is_err());
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
