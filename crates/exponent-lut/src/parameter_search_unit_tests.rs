    use super::*;

    fn reviewed_estimator_bits(universe: usize, q_l: usize) -> u64 {
        match (universe, q_l) {
            (450, 16) => 99,
            (451, 16) => 100,
            other => panic!("unexpected reviewed tuple {other:?}"),
        }
    }

    fn mock_prepared(candidate: Candidate) -> PreparedCandidate {
        PreparedCandidate {
            candidate,
            ring_dimension: 1 << candidate.log_ring_dimension,
            bucket_width: 3,
            official_preimage_bound: 10.into(),
            layout_id: [candidate.crt_depth as u8; 32],
            program_id: [candidate.log_ring_dimension as u8; 32],
            pbc_attempts_used: 1,
            mask_base_p_digit_count: 1,
            fresh_error_base_p_digit_count: 1,
            prf_component_count: 2,
            prf_coefficient_count: 1 << candidate.log_ring_dimension,
            prf_active_count: 1,
            prf_label_count: 1,
            prf_value_count: 1,
            bundle: None,
            average_report: None,
        }
    }

    fn mock_sparse_profile(config: &SearchConfig) -> SelectedSparseLwrProfile {
        let tuple = config.sparse_lwr_phase1_grid[0].clone();
        let (error_lower, error_upper) = sparse_lwr_error_bounds(tuple.q_l, tuple.p).unwrap();
        SelectedSparseLwrProfile {
            tuple: tuple.clone(),
            parameter_grid: config.sparse_lwr_phase1_grid.clone(),
            q_l: config.sparse_lwr_modulus,
            p: config.sparse_lwr_output_modulus,
            universe: config.sparse_lwr_universe,
            weight: config.sparse_lwr_weight,
            tier: SparseLwrSecurityTier::Primary100,
            error_lower,
            error_upper,
            estimator_commit: config.phase1_estimator_commit.clone(),
            estimator_cost_model: config.phase1_estimator_cost_model.clone(),
            estimator_shape_model: config.phase1_estimator_shape_model.clone(),
            sparse_lwr_security_bits: 200,
            raw_key_entropy_bits: raw_key_entropy_bits(
                config.sparse_lwr_universe,
                config.sparse_lwr_weight,
            ),
            evaluations: Vec::new(),
        }
    }

    #[test]
    fn grid_is_lexicographic_depth_then_ring_dimension_when_base_is_fixed() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 2..=3;
        config.log_ring_dimensions = 5..=6;
        config.crt_base_bits_grid = vec![(32, 16)];
        assert_eq!(
            candidates(&config).collect::<Vec<_>>(),
            vec![
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 2, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 3, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 3, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 },
            ]
        );
    }

    #[test]
    fn reviewed_grid_declares_phase_two_minimality_scope() {
        let config = SearchConfig::reviewed();
        assert_eq!(config.crt_depths, 30..=62);
        assert_eq!(config.log_ring_dimensions, 16..=17);
        assert_eq!(config.crt_bits, 32);
        assert_eq!(config.base_bits, 16);
        assert_eq!(config.crt_base_bits_grid, descending_crt_base_bits_grid(32));
        assert_eq!(config.crt_base_bits_grid[0], (32, 16));
    }

    #[test]
    fn crt_base_grid_descends_base_width_for_each_crt_width() {
        let grid = descending_crt_base_bits_grid(6);
        assert_eq!(grid, vec![(6, 3), (6, 2), (6, 1)]);
    }

    #[test]
    fn phase_two_candidates_keep_depth_ring_order_and_descend_base_bits() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=1;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(5, 2), (4, 2), (5, 1), (4, 1)];
        assert_eq!(
            candidates(&config).collect::<Vec<_>>(),
            vec![
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 5, base_bits: 2 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 4, base_bits: 2 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 5, base_bits: 1 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 4, base_bits: 1 },
            ]
        );
    }

    #[test]
    fn phase_two_candidates_exhaust_largest_base_before_smaller_base() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=2;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(32, 16), (30, 15), (32, 15)];
        let candidates = candidates(&config).collect::<Vec<_>>();
        assert_eq!(
            candidates,
            vec![
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
            ]
        );
    }

    #[test]
    fn average_selector_uses_minimum_crt_spacing_not_smallest_tower() {
        let spacing =
            minimum_crt_spacing(&[BigUint::from(2u8), BigUint::from(3u8), BigUint::from(5u8)])
                .unwrap();
        assert_eq!(spacing, BigUint::from(6u8));
        assert!(spacing > BigUint::from(2u8));
    }

    #[test]
    fn average_policy_is_derived_once_from_search_config() {
        let mut config = SearchConfig::reviewed();
        config.security_bits = 37;
        let policy = average_case_config(&config).unwrap();
        assert_eq!(policy.failure_exponent, 37);
        assert_eq!(policy.input_domain_log2, 0);
        assert!(policy.allow_average_acceptance);
        assert_eq!(policy, average_case_config(&config).unwrap());
    }

    #[test]
    fn average_boundary_enumerates_only_strictly_valid_doubled_masks() {
        let base = BigUint::from(2u8);
        let spacing = BigUint::from(6u8);
        let valid = (1..=4)
            .map(|digits| (digits, base.pow(digits) - BigUint::one()))
            .take_while(|(_, d2)| d2 * 2u8 < spacing.clone() * 2u8)
            .map(|(digits, _)| digits)
            .collect::<Vec<_>>();
        assert_eq!(valid, vec![1, 2]);
        assert!(base.pow(3) - BigUint::one() >= spacing);
    }

    #[test]
    fn phase_two_skips_qbit_overflow_per_crt_base_pair() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 62..=62;
        config.log_ring_dimensions = 16..=16;
        config.crt_base_bits_grid = vec![(32, 16), (30, 15), (28, 14)];
        assert_eq!(
            candidates(&config).collect::<Vec<_>>(),
            vec![
                Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 30, base_bits: 15 },
                Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 28, base_bits: 14 },
            ]
        );
        config.crt_depths = 63..=63;
        config.crt_base_bits_grid = vec![(32, 16)];
        assert!(candidates(&config).next().is_none());
    }

    #[test]
    fn lookup_overlay_applies_the_configured_tiny_profile() {
        let values = std::collections::BTreeMap::from([
            ("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", "1"),
            ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE_GRID", "4"),
            ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", "1"),
            ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", "4"),
            ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", "2"),
            ("MXX_EXPONENT_LUT_REFRESH_LUT_WIDTH", "8"),
            ("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", "4"),
            ("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", "2"),
            ("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH", "1"),
            ("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH", "1"),
            ("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION", "5"),
            ("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION", "5"),
        ]);
        let config =
            SearchConfig::from_lookup(|name| Ok(values.get(name).map(|value| (*value).to_owned())))
                .unwrap();
        assert_eq!(config.security_bits, 1);
        assert_eq!(config.sparse_lwr_universe_grid, vec![4]);
        assert_eq!(config.sparse_lwr_weight, 1);
        assert_eq!(config.sparse_lwr_modulus, 4);
        assert_eq!(config.sparse_lwr_output_modulus, 2);
        assert_eq!(config.lut_width, 8);
        assert_eq!(config.crt_bits, 4);
        assert_eq!(config.base_bits, 2);
        assert_eq!(config.crt_depths, 1..=1);
        assert_eq!(config.log_ring_dimensions, 5..=5);
    }

    #[test]
    fn lookup_overlay_keeps_unset_values_from_reviewed_profile() {
        let config = SearchConfig::from_lookup(|name| {
            Ok((name == "MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS").then(|| "1".to_owned()))
        })
        .unwrap();
        assert_eq!(config.security_bits, 1);
        assert_eq!(config.crt_bits, SearchConfig::reviewed().crt_bits);
        assert_eq!(config.base_bits, SearchConfig::reviewed().base_bits);
        assert_eq!(
            config.sparse_lwr_universe_grid,
            SearchConfig::reviewed().sparse_lwr_universe_grid
        );
    }

    #[test]
    fn lookup_overlay_rejects_invalid_ranges() {
        let invalid_range = std::collections::BTreeMap::from([
            ("MXX_EXPONENT_LUT_REFRESH_MIN_CRT_DEPTH", "2"),
            ("MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH", "1"),
        ]);
        let error = SearchConfig::from_lookup(|name| {
            Ok(invalid_range.get(name).map(|value| (*value).to_owned()))
        })
        .unwrap_err();
        assert!(error.contains("CRT depth range"));
    }

    #[test]
    fn lookup_overlay_propagates_non_unicode_lookup_failure() {
        let error = SearchConfig::from_lookup(|name| {
            if name == "MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS" {
                Err("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS must contain valid UTF-8".to_owned())
            } else {
                Ok(None)
            }
        })
        .unwrap_err();
        assert!(error.contains("valid UTF-8"));
    }

    #[test]
    fn mocked_security_and_checker_select_first_minimal_candidate() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=3;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(32, 16)];
        let sparse_profile = mock_sparse_profile(&config);
        let result = search_with_hooks(
            &config,
            &sparse_profile,
            |candidate| Ok(mock_prepared(candidate)),
            |candidate| Ok(candidate.crt_depth as u64 * 100),
            |prepared| Ok(prepared.candidate.crt_depth >= 2),
        )
        .unwrap();
        assert_eq!(
            result.candidate,
            Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 }
        );
    }

    #[test]
    fn phase_one_exhausts_ordered_grid_before_selecting_primary_profile() {
        let mut config = SearchConfig::reviewed();
        config.security_bits = 100;
        let mut calls = Vec::new();
        let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            calls.push(universe);
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        assert_eq!(calls, vec![450, 451]);
        assert_eq!(selected.universe, 451);
        assert_eq!(selected.q_l, 16);
        assert_eq!(selected.tier, SparseLwrSecurityTier::Primary100);
        assert_eq!(selected.sparse_lwr_security_bits, 100);
    }

    #[test]
    fn phase_one_rejects_an_estimator_result_that_does_not_match_declared_evidence() {
        let config = SearchConfig::reviewed();
        let error = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(128)).unwrap_err();
        assert!(error.contains("expected declared"));
    }

    #[test]
    fn qualified_profiles_preserve_all_primary_rows_for_phase_two_advancement() {
        let config = SearchConfig::reviewed();
        let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let profiles = qualified_sparse_lwr_profiles(&selected);
        assert_eq!(profiles.iter().map(|profile| profile.q_l).collect::<Vec<_>>(), [16]);
        assert!(profiles.iter().all(|profile| profile.tier == SparseLwrSecurityTier::Primary100));
    }

    #[test]
    fn phase_two_caches_sparse_security_and_defers_prepare_until_bgg_floor() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=1;
        config.log_ring_dimensions = 5..=6;
        config.crt_base_bits_grid = vec![(32, 16)];
        let sparse_profile = mock_sparse_profile(&config);
        let mut security_calls = Vec::new();
        let mut prepare_calls = Vec::new();
        let mut checker_calls = Vec::new();
        let result = search_with_hooks(
            &config,
            &sparse_profile,
            |candidate| {
                prepare_calls.push(candidate);
                Ok(mock_prepared(candidate))
            },
            |candidate| {
                security_calls.push(candidate);
                Ok(if candidate.log_ring_dimension == 5 { 99 } else { 100 })
            },
            |prepared| {
                checker_calls.push(prepared.candidate);
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(security_calls.len(), 2);
        assert_eq!(
            prepare_calls,
            vec![Candidate { crt_depth: 1, log_ring_dimension: 6, crt_bits: 32, base_bits: 16 }]
        );
        assert_eq!(checker_calls, prepare_calls);
        assert_eq!(result.sparse_lwr_universe, config.sparse_lwr_universe);
        assert_eq!(result.sparse_lwr_weight, config.sparse_lwr_weight);
    }

    #[test]
    fn phase_two_skips_exactly_infeasible_preparation_but_propagates_fatal_errors() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 1..=2;
        config.log_ring_dimensions = 5..=5;
        config.crt_base_bits_grid = vec![(32, 16)];
        let sparse_profile = mock_sparse_profile(&config);
        let mut prepared_candidates = Vec::new();
        let result = search_with_hooks(
            &config,
            &sparse_profile,
            |candidate| {
                prepared_candidates.push(candidate);
                if candidate.crt_depth == 1 {
                    Err(CandidatePreparationError::Infeasible(
                        "strict rounding is infeasible".to_owned(),
                    ))
                } else {
                    Ok(mock_prepared(candidate))
                }
            },
            |_| Ok(128),
            |_| Ok(true),
        )
        .unwrap();
        assert_eq!(
            prepared_candidates,
            vec![
                Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
                Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
            ]
        );
        assert_eq!(
            result.candidate,
            Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 }
        );

        let fatal = search_with_hooks(
            &config,
            &sparse_profile,
            |_| Err(CandidatePreparationError::Fatal("invalid setup identity".to_owned())),
            |_| Ok(128),
            |_| Ok(true),
        )
        .unwrap_err();
        assert_eq!(fatal, "invalid setup identity");
    }

    #[test]
    fn phase_one_rejects_a_non_ascending_grid() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_phase1_grid[1] = config.sparse_lwr_phase1_grid[0].clone();
        let error = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(200)).unwrap_err();
        assert!(error.contains("tuple grid") || error.contains("p == 2"));
    }

    fn checkpoint_test_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mxx-exponent-lut-{name}-{}.json", std::process::id()))
    }

    #[test]
    fn phase_one_checkpoint_reuses_an_exact_declaration() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_phase1_grid = config.sparse_lwr_phase1_grid[..2].to_vec();
        let path = checkpoint_test_path("phase1-reuse");
        let _ = std::fs::remove_file(&path);

        let mut first_calls = 0;
        let first = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            first_calls += 1;
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();

        let mut second_calls = 0;
        let second = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            second_calls += 1;
            Err("an exact checkpoint must not rerun Phase 1".to_owned())
        })
        .unwrap();
        assert_eq!(first_calls, 2);
        assert_eq!(second_calls, 0);
        assert_eq!(first.universe, second.universe);
        assert_eq!(second.universe, 451);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn phase_one_checkpoint_rejects_mismatch_and_missing_path_starts_fresh_search() {
        let mut config = SearchConfig::reviewed();
        config.sparse_lwr_phase1_grid = config.sparse_lwr_phase1_grid[..2].to_vec();
        let path = checkpoint_test_path("phase1-mismatch");
        let _ = std::fs::remove_file(&path);

        let mut calls = 0;
        let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            calls += 1;
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(selected.universe, 451);

        let mut changed = config.clone();
        changed.phase1_estimator_cost_model = "ALT".to_owned();
        let error = load_or_search_phase1(&changed, Some(&path), |_, _, _, _| {
            Err("a declaration mismatch must not fall back to a fresh search".to_owned())
        })
        .unwrap_err();
        assert!(error.contains("does not match"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn phase_one_checkpoint_binds_tuple_tier_error_and_estimator_identity() {
        let config = SearchConfig::reviewed();
        let path = checkpoint_test_path("phase1-identity");
        let _ = std::fs::remove_file(&path);
        load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let mut changed = config.clone();
        changed.sparse_lwr_phase1_grid[0].q_l = 10;
        assert!(
            load_or_search_phase1(&changed, Some(&path), |universe, _, q_l, _| Ok(
                reviewed_estimator_bits(universe, q_l)
            ))
            .unwrap_err()
            .contains("does not match")
        );
        let mut checkpoint: Phase1Checkpoint =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        checkpoint.selected.tier = SparseLwrSecurityTier::Fallback100;
        std::fs::write(&path, serde_json::to_vec_pretty(&checkpoint).unwrap()).unwrap();
        assert!(
            load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| Ok(
                reviewed_estimator_bits(universe, q_l)
            ))
            .unwrap_err()
            .contains("final row")
        );
        std::fs::remove_file(&path).unwrap();
        load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let mut changed = config.clone();
        changed.phase1_estimator_shape_model = "primal".to_owned();
        assert!(
            load_or_search_phase1(&changed, Some(&path), |universe, _, q_l, _| Ok(
                reviewed_estimator_bits(universe, q_l)
            ))
            .unwrap_err()
            .contains("does not match")
        );
        let mut changed = config.clone();
        changed.phase1_fallback_security_bits = 101;
        assert!(
            load_or_search_phase1(&changed, Some(&path), |_, _, _, _| Ok(128))
                .unwrap_err()
                .contains("does not match")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn phase_one_checkpoint_round_trips_the_boundary_tuple() {
        for selected_universe in [451] {
            let mut config = SearchConfig::reviewed();
            config.sparse_lwr_phase1_grid = config
                .sparse_lwr_phase1_grid
                .into_iter()
                .filter(|tuple| tuple.nu == selected_universe)
                .collect();
            let path = checkpoint_test_path(&format!("phase1-selected-{selected_universe}"));
            let _ = std::fs::remove_file(&path);
            let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
                Ok(reviewed_estimator_bits(universe, q_l))
            })
            .unwrap();
            assert_eq!(selected.universe, selected_universe);
            let reused = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
                Err("an exact checkpoint must not rerun Phase 1".to_owned())
            })
            .unwrap();
            assert_eq!(reused.tuple, selected.tuple);
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn accepted_phase_two_profile_persists_q16_and_rejects_out_of_grid_tampering() {
        let config = SearchConfig::reviewed();
        let path = checkpoint_test_path("phase2-accepted");
        let _ = std::fs::remove_file(&path);
        let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let mut order = Vec::new();
        let (q16, result) = search_qualified_profiles(&selected, |profile| {
            order.push(profile.q_l);
            Ok(SearchResult {
                candidate: Candidate {
                    crt_depth: 30,
                    log_ring_dimension: 16,
                    crt_bits: 32,
                    base_bits: 16,
                },
                achieved_security_bits: 128,
                bgg_rlwe_security_bits: 128,
                sparse_lwr_security_bits: profile.sparse_lwr_security_bits,
                raw_key_entropy_bits: profile.raw_key_entropy_bits,
                sparse_lwr_universe: profile.universe,
                sparse_lwr_weight: profile.weight,
                official_preimage_bound: "1".to_owned(),
                ring_dimension: 1 << 16,
                bucket_width: 1,
                pbc_attempts_used: 1,
                layout_id: "aa".repeat(32),
                program_id: "bb".repeat(32),
                checker_accepted: true,
                sparse_lwr_q_l: profile.q_l,
                sparse_lwr_p: profile.p,
                sparse_lwr_tier: profile.tier,
                estimator_commit: profile.estimator_commit.clone(),
                estimator_cost_model: profile.estimator_cost_model.clone(),
                estimator_shape_model: profile.estimator_shape_model.clone(),
                average_evidence: None,
            })
        })
        .unwrap();
        assert_eq!(order, vec![16]);
        assert_eq!(q16.q_l, 16);
        let candidate = result.candidate;
        persist_accepted_phase2_profile(&path, &config, &q16, candidate).unwrap();
        let checkpoint: Phase1Checkpoint =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        checkpoint.validate(&config).unwrap();
        assert_eq!(checkpoint.accepted_phase2.as_ref().unwrap().tuple.q_l, 16);
        assert_eq!(checkpoint.accepted_phase2.as_ref().unwrap().candidate, candidate);

        let mut tampered = checkpoint;
        tampered.accepted_phase2.as_mut().unwrap().candidate.crt_bits = 31;
        std::fs::write(&path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(
            Phase1Checkpoint::validate(&tampered, &config)
                .unwrap_err()
                .contains("outside the declared search grid")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn profile_advancement_propagates_noncanonical_errors() {
        let config = SearchConfig::reviewed();
        let selected = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let error = search_qualified_profiles(&selected, |_| Err("fatal setup error".to_owned()))
            .unwrap_err();
        assert_eq!(error, "fatal setup error");
        let error = search_qualified_profiles(&selected, |_| {
            Err(format!("{NO_CANDIDATE_ERROR}: extra diagnostic"))
        })
        .unwrap_err();
        assert_eq!(error, format!("{NO_CANDIDATE_ERROR}: extra diagnostic"));
    }

    #[test]
    fn malformed_phase_one_checkpoint_fails_before_security_callback() {
        let config = SearchConfig::reviewed();
        let path = checkpoint_test_path("phase1-malformed");
        std::fs::write(&path, b"{not valid json").unwrap();

        let mut calls = 0;
        let error = load_or_search_phase1(&config, Some(&path), |_, _, _, _| {
            calls += 1;
            Ok(129)
        })
        .unwrap_err();
        assert!(error.contains("read Phase-1 checkpoint"));
        assert_eq!(calls, 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn result_serialization_is_redacted() {
        let result = SearchResult {
            candidate: Candidate {
                crt_depth: 2,
                log_ring_dimension: 5,
                crt_bits: 32,
                base_bits: 16,
            },
            achieved_security_bits: 128,
            bgg_rlwe_security_bits: 130,
            sparse_lwr_security_bits: 140,
            raw_key_entropy_bits: raw_key_entropy_bits(512, 64),
            sparse_lwr_universe: 512,
            sparse_lwr_weight: 64,
            official_preimage_bound: "123".to_owned(),
            ring_dimension: 32,
            bucket_width: 4,
            pbc_attempts_used: 1,
            layout_id: "aa".repeat(32),
            program_id: "bb".repeat(32),
            checker_accepted: true,
            sparse_lwr_q_l: 8,
            sparse_lwr_p: 2,
            sparse_lwr_tier: SparseLwrSecurityTier::Primary100,
            estimator_commit: REVIEWED_ESTIMATOR_COMMIT.to_owned(),
            estimator_cost_model: REVIEWED_ESTIMATOR_COST_MODEL.to_owned(),
            estimator_shape_model: REVIEWED_ESTIMATOR_SHAPE_MODEL.to_owned(),
            average_evidence: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("support"));
        assert!(!json.contains("selected"));
        assert!(json.contains("official_preimage_bound"));
    }

    #[test]
    fn persisted_report_declares_minimality_scope_and_evaluated_phase_one_rows() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 2..=3;
        config.log_ring_dimensions = 5..=6;
        let sparse_profile = select_sparse_lwr_profile(&config, |universe, _, q_l, _| {
            Ok(reviewed_estimator_bits(universe, q_l))
        })
        .unwrap();
        let result = SearchResult {
            candidate: Candidate {
                crt_depth: 2,
                log_ring_dimension: 5,
                crt_bits: 32,
                base_bits: 16,
            },
            achieved_security_bits: 128,
            bgg_rlwe_security_bits: 128,
            sparse_lwr_security_bits: sparse_profile.sparse_lwr_security_bits,
            raw_key_entropy_bits: sparse_profile.raw_key_entropy_bits,
            sparse_lwr_universe: sparse_profile.universe,
            sparse_lwr_weight: sparse_profile.weight,
            official_preimage_bound: "123".to_owned(),
            ring_dimension: 32,
            bucket_width: 4,
            pbc_attempts_used: 1,
            layout_id: "aa".repeat(32),
            program_id: "bb".repeat(32),
            checker_accepted: true,
            sparse_lwr_q_l: sparse_profile.q_l,
            sparse_lwr_p: sparse_profile.p,
            sparse_lwr_tier: sparse_profile.tier,
            estimator_commit: sparse_profile.estimator_commit.clone(),
            estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
            estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
            average_evidence: None,
        };
        let json = serde_json::to_string(&search_report(&config, &sparse_profile, result)).unwrap();

        assert!(json.contains("\"parameter_grid\":["));
        assert!(json.contains("\"universe_grid\":[451]"));
        assert!(json.contains(&format!("\"support_weight\":{}", sparse_profile.weight)));
        assert!(json.contains("\"q_l\":16"));
        assert!(json.contains("\"output_modulus\":2"));
        assert!(json.contains("\"lut_width\":512"));
        assert!(json.contains("\"security_target_bits\":100"));
        assert!(json.contains("\"sparse_secret_model\":\"SparseBinary\""));
        assert!(json.contains("\"sparse_error_model\":\"Uniform\""));
        assert!(json.contains("\"selected_error_lower\":-4"));
        assert!(json.contains("\"selected_error_upper\":3"));
        assert!(json.contains("\"exact_estimator\":true"));
        assert!(json.contains("\"evaluated\":["));
        assert!(json.contains("\"universe\":451"));
        assert!(json.contains("\"minimum_security_bits\""));
        assert!(json.contains("\"qualified\":true"));
        assert!(json.contains("\"selected_universe\":451"));
        assert!(json.contains("\"crt_depth_min\":2"));
        assert!(json.contains("\"crt_depth_max\":3"));
        assert!(json.contains("\"log_ring_dimension_min\":5"));
        assert!(json.contains("\"log_ring_dimension_max\":6"));
        assert!(
            json.contains("\"order\":\"base_bits_descending_then_crt_bits_descending_then_crt_depth_then_log_ring_dimension\"")
        );
        assert!(json.contains("\"crt_base_bits_grid\":[[32,16]"));
        assert!(!json.contains("selected_slots"));
        assert!(!json.contains("support_coordinates"));
    }

    #[test]
    fn sparse_lwr_estimator_arguments_use_exact_reviewed_model() {
        assert_eq!(sparse_lwr_error_bounds(16, 2).unwrap(), (-4, 3));
        assert_eq!(
            sparse_lwr_estimator_args(512, 64, 16, 2).unwrap(),
            vec![
                "512",
                "16",
                "--s-dist",
                r#"{"name":"SparseBinary","hw":64,"n":512}"#,
                "--e-dist",
                r#"{"name":"Uniform","a":-4,"b":3}"#,
                "--exact",
            ]
        );
        assert!(sparse_lwr_error_bounds(15, 2).is_err());
        assert!(sparse_lwr_error_bounds(16, 0).is_err());
    }

    #[test]
    fn reconstruction_coefficients_come_from_the_dcrt_parameters() {
        let dcrt = DCRTPolyParams::new(32, 2, 28, 27);
        let coefficients = exact_reconstruction_coefficients(&dcrt);
        assert_eq!(coefficients.len(), 2);
        assert!(dcrt.reconst_coeffs().iter().any(|value| value != &BigUint::from(1_u8)));
    }

    #[test]
    fn security_output_parser_is_strict_and_fail_closed() {
        assert_eq!(parse_security_bits(b"128\n"), Ok(128));
        assert!(parse_security_bits(b"\n128\n\n").is_ok());
        assert_eq!(parse_security_bits(b"Algorithm ... failed\nalgorithm-result\n128\n"), Ok(128));
        assert!(parse_security_bits(b"128\n129\n").is_err());
        assert!(parse_security_bits(b"security: 128\n").is_err());
        assert!(parse_security_bits(b"+128\n").is_err());
        assert!(parse_security_bits(b"18446744073709551616\n").is_err());
    }

    #[test]
    fn fresh_digit_selection_is_strict_and_finite() {
        assert_eq!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[32, 37]), Ok(1));
        assert!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[31, 32]).is_err());
    }

    #[test]
    fn mask_digit_selection_checks_domain_hiding_and_rounding_together() {
        // This small profile has an exact finite answer. The first digit
        // covers Q_L but cannot meet the joint hiding requirement; the second
        // digit is rejected by strict rounding, so the selector fails closed.
        let result = select_mask_base_p_digit_count(
            &BigUint::from(2u8),
            &BigUint::from(2u8),
            &BigUint::from(15u8),
            &[BigUint::from(3u8), BigUint::from(5u8)],
            1,
            1,
            &BigUint::from(2u8),
            &BigUint::one(),
            &BigUint::one(),
            &BigUint::zero(),
            &BigUint::zero(),
            0,
            1,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn mask_digit_selection_returns_the_first_digit_after_hiding_threshold() {
        // The two CRT towers require ell_beta=14. d_m=1 gives M_m=64 and
        // covers Q_L=2, but the exact joint hiding requirement is 3584.
        // d_m=2 gives M_m=4096 and satisfies the strict rounding bound (the
        // smallest spacing is 9001).
        let q = BigUint::from(9_001u16) * BigUint::from(9_007u16);
        let result = select_mask_base_p_digit_count(
            &BigUint::from(64u8),
            &BigUint::from(2u8),
            &q,
            &[BigUint::from(9_001u16), BigUint::from(9_007u16)],
            1,
            14,
            &BigUint::from(4u8),
            &BigUint::one(),
            &BigUint::one(),
            &BigUint::zero(),
            &BigUint::zero(),
            0,
            1,
            1,
        )
        .unwrap();
        assert_eq!(result, 2);
    }

    #[test]
    fn mask_digit_selection_rejects_incomplete_or_non_coprime_crt() {
        let args = |full_modulus: BigUint, crt_moduli: Vec<BigUint>| {
            select_mask_base_p_digit_count(
                &BigUint::from(8u8),
                &BigUint::from(2u8),
                &full_modulus,
                &crt_moduli,
                1,
                1,
                &BigUint::from(2u8),
                &BigUint::one(),
                &BigUint::zero(),
                &BigUint::zero(),
                &BigUint::one(),
                0,
                1,
                1,
            )
        };
        assert!(args(BigUint::from(12u8), vec![BigUint::from(2u8), BigUint::from(3u8)]).is_err());
        assert!(
            args(
                BigUint::from(12u8),
                vec![BigUint::from(2u8), BigUint::from(2u8), BigUint::from(3u8)]
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "diagnostic-only large exact-noise profile"]
    fn emits_large_profile_sparse_prf_stage_diagnostics() {
        let mut config = SearchConfig::reviewed();
        config.crt_depths = 62..=62;
        config.log_ring_dimensions = 16..=16;
        config.crt_bits = 32;
        config.base_bits = 1;
        let ring_dimension = 1usize << 16;
        let dcrt = DCRTPolyParams::new(ring_dimension as u32, 62, 32, 1);
        let q_bits = dcrt.modulus().bits();
        assert!(q_bits < 2_000, "diagnostic profile q bits = {q_bits}");
        info!(q_bits, ring_dimension, "large exact-noise diagnostic profile");
        let result = prepare_candidate(
            &config,
            Candidate { crt_depth: 62, log_ring_dimension: 16, crt_bits: 32, base_bits: 1 },
        );
        info!(accepted = result.is_ok(), "large exact-noise diagnostic completed");
    }

    #[test]
    fn diag_free_prepare_and_check_bundle_smoke() {
        let config = SearchConfig::reviewed();
        let result = prepare_candidate(
            &config,
            Candidate { crt_depth: 30, log_ring_dimension: 15, crt_bits: 32, base_bits: 16 },
        );
        if let Ok(mut prepared) = result {
            let checked = check_refresh_bundle(&config, prepared.candidate, &mut prepared);
            assert!(checked.is_ok(), "bundle checker returned {checked:?}");
        }
    }
