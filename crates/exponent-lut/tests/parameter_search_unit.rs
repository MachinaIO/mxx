include!("test_utils.rs");

fn evaluate_universal_depth_case(
    config: &SearchConfig,
    directory: &Path,
    depth: usize,
    dropped: usize,
) -> serde_json::Value {
    let mut config = config.clone();
    config.gadget_dropped_moduli = dropped;
    let candidate =
        Candidate { crt_depth: depth, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
    let result = (|| -> Result<serde_json::Value, CandidatePreparationError> {
        let mut context = build_candidate_context(&config, candidate)?;
        context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, 5)?;
        let prepared =
            prepare_candidate_from_ring_context(&config, &context, Some((38, 1)), false)?;
        let snapshot = prepared.noise_snapshot.as_ref().expect("production snapshot");
        let average = snapshot.simulate_average(&average_case_config(&config).unwrap()).unwrap();
        let row = serde_json::json!({
            "depth":depth,"dropped":dropped,"retained":depth-dropped,
            "gadget_columns":4*(depth-dropped),"accepted":average.accepted,
            "full_modulus":snapshot.refresh.global_modulus.as_ref().unwrap().to_string(),
            "crt_moduli":context.crt_moduli,
            "switch_moduli":context.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "max_f_avg":average.refresh.max_f_avg.to_string(),
            "minimum_rounding_margin":average.refresh.slots.iter().map(|s| &s.rounding_margin).min().unwrap().to_string(),
            "minimum_smudging_margin":average.refresh.slots.iter().map(|s| &s.smudging_margin).min().unwrap().to_string(),
            "route_bias_bounds":snapshot.refresh.route_bias_bounds.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "average":average,
            "setup_identity":hex(*snapshot.setup_identity()),
        });
        Ok(row)
    })();
    let row = match result {
        Ok(row) => row,
        Err(error) => serde_json::json!({"depth":depth,"dropped":dropped,
                "accepted":false,"preparation_error":error.to_string()}),
    };
    fs::write(
        directory.join(format!("depth-{depth}-drop-{dropped}.json")),
        serde_json::to_vec_pretty(&row).unwrap(),
    )
    .unwrap();
    eprintln!(
        "DEPTH_WINDOW_CASE depth={depth} dropped={dropped} accepted={} F={} error={}",
        row["accepted"], row["max_f_avg"], row["preparation_error"]
    );
    row
}

#[test]
#[ignore = "explicit CPU-only exact-depth baseline and bounded universal approximation sweep"]
fn universal_approximate_minimum_depth_window() {
    use rayon::prelude::*;
    let mut config = SearchConfig::reviewed();
    config.noise_model = NoiseSearchModel::AverageCase;
    config.sparse_lwr_universe = 8192;
    config.sparse_lwr_weight = 18;
    config.sparse_lwr_modulus = 4096;
    config.sparse_lwr_output_modulus = 32;
    let run =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/universal_approximate_depth_window")
        .join(run.to_string());
    fs::create_dir_all(&directory).unwrap();
    eprintln!("DEPTH_WINDOW_DIRECTORY {}", directory.display());
    let evaluate =
        |depth, dropped| evaluate_universal_depth_case(&config, &directory, depth, dropped);
    let mut baseline = (1..5)
        .map(|depth| {
            serde_json::json!({
                "depth":depth,"dropped":0,"accepted":false,
                "exclusion":"fixed exact decoder switch width five exceeds full CRT depth"
            })
        })
        .collect::<Vec<_>>();
    let mut minimum = None;
    for depth in 5..=33 {
        let row = evaluate(depth, 0);
        let accepted = row["accepted"] == true;
        baseline.push(row);
        if accepted {
            minimum = Some(depth);
            break;
        }
    }
    let minimum = minimum.expect("exact accepted baseline inside conventional security cap");
    assert!(minimum + 3 <= 33, "depth window extends beyond verified security envelope");
    let candidates = (minimum..=minimum + 3)
        .flat_map(|depth| (0..depth).map(move |drop| (depth, drop)))
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    // Four concurrent independent snapshots cap peak program-state memory.
    for chunk in candidates.chunks(4) {
        rows.extend(
            chunk.par_iter().map(|&(depth, drop)| evaluate(depth, drop)).collect::<Vec<_>>(),
        );
    }
    let best = rows
        .iter()
        .filter(|row| row["accepted"] == true)
        .map(|row| row["dropped"].as_u64().unwrap())
        .max()
        .unwrap();
    // Exact failure does not imply approximate failure: separately exclude
    // every lower-depth configuration that could improve the incumbent.
    // Depth <= best+1 cannot retain even one tower while dropping more.
    let lower_candidates = (5.max(best as usize + 2)..minimum)
        .flat_map(|depth| ((best as usize + 1)..depth).map(move |drop| (depth, drop)))
        .collect::<Vec<_>>();
    let mut lower_rows = Vec::new();
    for chunk in lower_candidates.chunks(4) {
        lower_rows.extend(
            chunk.par_iter().map(|&(depth, drop)| evaluate(depth, drop)).collect::<Vec<_>>(),
        );
    }
    assert!(
        baseline
            .iter()
            .chain(&rows)
            .chain(&lower_rows)
            .all(|row| row.get("preparation_error").is_none()),
        "unexpected preparation errors require explicit audit, not rejection by default"
    );
    let best = rows
        .iter()
        .chain(&lower_rows)
        .filter(|row| row["accepted"] == true)
        .map(|row| row["dropped"].as_u64().unwrap())
        .max()
        .unwrap();
    let winners = rows
        .iter()
        .chain(&lower_rows)
        .filter(|row| row["accepted"] == true && row["dropped"] == best)
        .map(|row| serde_json::json!({"depth":row["depth"],"dropped":row["dropped"]}))
        .collect::<Vec<_>>();
    fs::write(directory.join("search.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "scope":"Current universal approximate production model; fixed N65536, CRT50/base25, dm38/df1, five-tower exact decoder; conventional security envelope through depth33; not a joint-security proof",
        "minimum_exact_depth":minimum,"maximum_dropped":best,"winners":winners,
        "baseline":baseline,"window":rows,"lower_depth_exclusions":lower_rows
    })).unwrap()).unwrap();
    eprintln!(
        "DEPTH_WINDOW_RESULT minimum_exact_depth={minimum} maximum_dropped={best} winners={winners:?}"
    );
}

#[test]
#[ignore = "explicit CPU-only universal approximate gadget noise and graph preparation"]
fn universal_approximate_complete_lower_depths() {
    use rayon::prelude::*;
    let mut config = SearchConfig::reviewed();
    config.noise_model = NoiseSearchModel::AverageCase;
    config.sparse_lwr_universe = 8192;
    config.sparse_lwr_weight = 18;
    config.sparse_lwr_modulus = 4096;
    config.sparse_lwr_output_modulus = 32;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/universal_approximate_depth_window");
    let directory = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.join("search.json").is_file())
        .max()
        .expect("completed first-stage sweep");
    let mut search: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join("search.json")).unwrap()).unwrap();
    assert!(
        search["lower_depth_exclusions"].is_array(),
        "first-stage maximum proof must be complete"
    );
    let minimum = search["minimum_exact_depth"].as_u64().unwrap() as usize;
    let best = search["maximum_dropped"].as_u64().unwrap() as usize;
    let candidates = (5..minimum)
        .flat_map(|depth| (1..=best.min(depth - 1)).map(move |drop| (depth, drop)))
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for chunk in candidates.chunks(4) {
        rows.extend(
            chunk
                .par_iter()
                .map(|&(depth, drop)| {
                    evaluate_universal_depth_case(&config, &directory, depth, drop)
                })
                .collect::<Vec<_>>(),
        );
    }
    assert!(rows.iter().all(|row| row.get("preparation_error").is_none()));
    let accepted = rows.iter().filter(|row| row["accepted"] == true).count();
    search["lower_depth_small_drops"] = serde_json::Value::Array(rows);
    fs::write(directory.join("complete-search.json"), serde_json::to_vec_pretty(&search).unwrap())
        .unwrap();
    eprintln!(
        "DEPTH_WINDOW_SUPPLEMENT directory={} cases={} accepted={accepted}",
        directory.display(),
        candidates.len()
    );
}

#[test]
#[ignore = "explicit CPU-only universal approximate gadget noise and graph preparation"]
fn universal_approximate_selected_noise() {
    export_selected_noise_and_graph(
        33,
        7,
        102,
        "universal_approximate_selected",
        "test_data/approximate_gadget_maximum_drop/1788656642909207371/mxx-approximate-n16-exact-security-cap.log",
    );
}

#[test]
#[ignore = "explicit CPU-only exact minimum-depth noise and benchmark graph export"]
fn exact_minimum_depth_selected_noise() {
    export_selected_noise_and_graph(
        26,
        0,
        143,
        "exact_minimum_depth_selected",
        "test_data/universal_approximate_depth_window/1788660855725121579/security-depth26.log",
    );
}

fn export_selected_noise_and_graph(
    depth: usize,
    dropped: usize,
    security: u64,
    output_name: &str,
    security_evidence: &str,
) {
    let mut config = SearchConfig::reviewed();
    config.noise_model = NoiseSearchModel::AverageCase;
    config.sparse_lwr_universe = 8192;
    config.sparse_lwr_weight = 18;
    config.sparse_lwr_modulus = 4096;
    config.sparse_lwr_output_modulus = 32;
    config.gadget_dropped_moduli = dropped;
    let candidate =
        Candidate { crt_depth: depth, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
    let mut context = build_candidate_context(&config, candidate).unwrap();
    context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, 5).unwrap();
    let prepared =
        prepare_candidate_from_ring_context(&config, &context, Some((38, 1)), false).unwrap();
    let snapshot = prepared.noise_snapshot.as_ref().unwrap();
    let average = snapshot.simulate_average(&average_case_config(&config).unwrap()).unwrap();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data").join(output_name);
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("noise.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "scope":"Universal approximate gadget, exact rectangular trapdoor preimages, even-corrected routes; AverageCase analytical acceptance only",
        "candidate":candidate,"gadget_dropped_moduli":dropped,"gadget_columns":4*(depth-dropped),
        "crt_moduli":context.crt_moduli,"switch_moduli":context.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "route_bias_bounds":snapshot.refresh.route_bias_bounds.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "average":average,"setup_identity":hex(*snapshot.setup_identity()),
        "layout_id":hex(prepared.layout_id),"program_id":hex(prepared.program_id),
    })).unwrap()).unwrap();
    eprintln!(
        "UNIVERSAL_APPROXIMATE_RESULT {} accepted={} max_F={}",
        directory.display(),
        average.accepted,
        average.refresh.max_f_avg
    );
    assert_eq!(context.sampler_layout.public_key_columns(), 4 * (depth - dropped));
    assert_eq!(snapshot.operation_residuals.is_empty(), dropped == 0);
    if dropped == 0 {
        assert!(snapshot.readout_residuals.is_none());
        assert!(snapshot.refresh.route_bias_bounds.iter().all(BigUint::is_zero));
    }
    assert!(average.accepted, "new universal approximate noise model must pass independently");
    fs::write(directory.join("benchmark-input.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "security_scope":"Conventional full-q RLWE rough estimate and pinned sparse LWR 103.186; correlated joint-view security not established",
        "security_evidence":security_evidence,
        "result":{
            "candidate":candidate,"gadget_dropped_moduli":dropped,"achieved_security_bits":security.min(103),
            "bgg_security_bits":security,
            "sparse_lwr_universe":8192,"sparse_lwr_weight":18,"sparse_lwr_q_l":4096,"sparse_lwr_p":32,
            "layout_id":hex(prepared.layout_id),"program_id":hex(prepared.program_id),
            "average_evidence":{
                "setup_identity":hex(*snapshot.setup_identity()),
                "switch_moduli":context.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "mask_digit_count":38,"fresh_error_digit_count":1,
                "hard_authority_accepted":average.hard_authority_accepted,
                "correctness_accepted":average.correctness_accepted,"accepted":average.accepted
            }
        }
    })).unwrap()).unwrap();
    let graph_prepared =
        prepare_candidate_with_fixed_digits(&config, candidate, 38, 1, &context.switch_moduli)
            .unwrap();
    let bundle = graph_prepared.bundle.as_ref().unwrap();
    assert_eq!(bundle.public_identity(), snapshot.setup_identity());
    let preprocessing = bundle.benchmark_preprocessing_graph();
    preprocessing
        .validate_with_manifests(
            &mxx_ir_core::ParamEnv::default(),
            bundle.preprocessing_manifests(),
        )
        .unwrap();
    bundle
        .benchmark_online_graph()
        .validate_with_manifests(&mxx_ir_core::ParamEnv::default(), bundle.verification_manifests())
        .unwrap();
    eprintln!(
        "UNIVERSAL_APPROXIMATE_GRAPH validated short ordinary gadget and exact rectangular decoder graph"
    );
}

#[test]
#[ignore = "explicit CPU-only approximate gadget noise model; no backend execution"]
fn profile_b_approximate_gadget_noise_simulation() {
    approximate_gadget_noise_cases(27, 16, 38, 5, false, None);
}

#[test]
#[ignore = "explicit CPU-only fixed-N maximum-drop pilot; no backend execution"]
fn approximate_gadget_maximum_drop_pilot() {
    let retained = (1..=33usize).collect::<Vec<_>>();
    for digits in [38usize, 39] {
        let rows = approximate_gadget_noise_cases(33, 16, digits, 5, true, Some(&retained));
        let selected = rows.iter().find(|row| row["modeled_refresh_accepted"] == true).unwrap();
        eprintln!(
            "MAXIMUM_DROP_PILOT dm={digits} dropped={} retained={} F_avg={}",
            selected["dropped_towers"],
            selected["retained_towers"],
            selected["conditional_refresh_max_F_avg"]
        );
    }
}

#[test]
#[ignore = "explicit CPU-only fixed-N maximum-drop proof grid; no backend execution"]
fn approximate_gadget_maximum_drop_frontier() {
    let mut config = SearchConfig::reviewed();
    config.noise_model = NoiseSearchModel::AverageCase;
    config.sparse_lwr_universe = 8192;
    config.sparse_lwr_weight = 18;
    config.sparse_lwr_modulus = 4096;
    config.sparse_lwr_output_modulus = 32;
    let policy = average_case_config(&config).unwrap();
    let run =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/approximate_gadget_maximum_drop")
        .join(run.to_string());
    fs::create_dir_all(&directory).unwrap();
    eprintln!("MAXIMUM_DROP_DIRECTORY {}", directory.display());
    let mut records = Vec::new();
    let mut unresolved = Vec::new();
    let max_mask = BigUint::from(32u8).pow(39);
    let ceil_div = |n: BigUint, d: &BigUint| (n + d - 1u8) / d;
    // Incumbent k=7 was independently found at depth33. Depth<=8 cannot
    // beat it since at least one tower must remain. This is a count bound,
    // not a monotonicity assumption about the noise model.
    for depth in 9..=33usize {
        let candidate =
            Candidate { crt_depth: depth, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
        let mut context = build_candidate_context(&config, candidate).unwrap();
        context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, 5).unwrap();
        let setup = setup_for_candidate_context(&config, &context, 3, 1).unwrap();
        let prepared =
            prepare_candidate_from_ring_context(&config, &context, Some((3, 1)), false).unwrap();
        let baseline = prepared.noise_snapshot.unwrap();
        for dropped in 8..depth {
            let retained = depth - dropped;
            let (snapshot, pre_refresh, _, _, _, _, _, _) =
                modeled_approximate_pre_refresh(&baseline, &context, retained, true);
            let average =
                snapshot.simulate_average_with_pre_refresh(&policy, pre_refresh.clone()).unwrap();
            let routes = snapshot.refresh.route_gains.as_ref().unwrap();
            let smudging = snapshot.refresh.smudging.as_ref().unwrap();
            let smudge_factor = (BigUint::one() << (smudging.lambda_mask + 1)) *
                BigUint::from(smudging.transcript_coordinates);
            // Drop every nonnegative mask error contribution, use the
            // smallest joint-tail event count (dm3) and largest M (dm39).
            // This is optimistic for every admissible mask digit count.
            let witness=average.refresh.slots.iter().find_map(|slot| {
                let variance=slot.variance_breakdown.readout_fresh.add(&slot.variance_breakdown.state).add(&slot.variance_breakdown.decoder);
                let lower=mxx_exponent_lut::noise_refresh::simulation::average_favg(&average.refresh.z_squared,&variance,0).unwrap()
                    +ceil_div(slot.rounding_numerator.clone(),&slot.rounding_denominator)
                    +ceil_div(&pre_refresh.readout_bias_bound*&routes.fresh[slot.slot].l1,&slot.lift);
                let smudge_lhs=&lower*&smudge_factor;
                let round_lhs=&lower*2u8;
                (smudge_lhs>max_mask || round_lhs>=slot.spacing).then(||serde_json::json!({"slot":slot.slot,"F_lower":lower.to_string(),"smudging_lhs":smudge_lhs.to_string(),"M_max":max_mask.to_string(),"rounding_lhs":round_lhs.to_string(),"spacing":slot.spacing.to_string()}))
            });
            if let Some(witness) = witness {
                records.push(serde_json::json!({"depth":depth,"dropped":dropped,"status":"optimistic_lower_bound_rejects_all_dm3_to39","witness":witness}));
            } else {
                let mut accepted = None;
                for digits in 3..=39 {
                    let gains = RefreshParameterSimulationRequest::mask_route_gain_for_digit_count(
                        &setup, digits,
                    )
                    .unwrap();
                    let changed = snapshot_with_mask_gain(&snapshot, digits, &gains).unwrap();
                    if changed
                        .average_refresh_accepted_from_pre_refresh(&policy, &pre_refresh)
                        .unwrap()
                    {
                        accepted = Some(digits);
                        break;
                    }
                }
                unresolved.push(
                    serde_json::json!({"depth":depth,"dropped":dropped,"accepted_digits":accepted}),
                );
                records.push(serde_json::json!({"depth":depth,"dropped":dropped,"status":"exhaustive_dm3_to39","accepted_digits":accepted}));
            }
        }
        fs::write(directory.join("frontier.json"), serde_json::to_vec_pretty(&records).unwrap())
            .unwrap();
        eprintln!(
            "MAXIMUM_DROP_DEPTH depth={depth} checked={} unresolved={}",
            records.len(),
            unresolved.len()
        );
    }
    assert!(
        unresolved.iter().all(|row| row["accepted_digits"].is_null()),
        "incumbent exceeded; inspect persisted grid"
    );
    fs::write(directory.join("search.json"),serde_json::to_vec_pretty(&serde_json::json!({"ring_dimension":65536,"security_depth_cap":33,"incumbent_dropped":7,"switch_width":5,"mask_digit_range":[3,39],"checked":records.len(),"exhaustive_fallback":unresolved,"records":records,"scope":"Maximum absolute dropped count within fixed50/25, fixed five-tower switching model and security-qualified depths; explicit analytical even-residual bridge, not runtime or joint-security proof"})).unwrap()).unwrap();
    eprintln!("MAXIMUM_DROP_COMPLETE {}", directory.display());
}

#[test]
#[ignore = "explicit CPU-only approximate gadget depth search; no backend execution"]
fn approximate_gadget_depth_search() {
    let config = SearchConfig::from_env().unwrap();
    let log_ring = *config.log_ring_dimensions.start();
    let started = std::time::Instant::now();
    let mut selected = [None, None];
    let mut rows = Vec::new();
    for depth in 27..=90 {
        for mask_digits in [38usize, 39] {
            let cases = approximate_gadget_noise_cases(depth, log_ring, mask_digits, 5, true, None);
            for (ratio, case) in [cases[1].clone(), cases[2].clone()].into_iter().enumerate() {
                if selected[ratio].is_none() && case["modeled_refresh_accepted"] == true {
                    selected[ratio] = Some(case.clone());
                    eprintln!(
                        "APPROXIMATE_DEPTH_FIRST_PASS ratio={ratio} depth={depth} dm={mask_digits} elapsed={:.3}",
                        started.elapsed().as_secs_f64()
                    );
                }
                rows.push(case);
            }
        }
        if selected.iter().all(Option::is_some) {
            break;
        }
    }
    let run =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/approximate_gadget_depth_search")
        .join(run.to_string());
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("search.json"),serde_json::to_vec_pretty(&serde_json::json!({"log_ring_dimension":log_ring,"selected":selected,"cases":rows,"elapsed_seconds":started.elapsed().as_secs_f64(),"scope":"AverageCase noise model with explicit even-residual bridge; not runtime or security acceptance"})).unwrap()).unwrap();
    eprintln!("APPROXIMATE_DEPTH_COMPLETE {}", directory.display());
}

fn modeled_approximate_pre_refresh(
    baseline: &ExponentLutNoiseSnapshot,
    context: &CandidatePreparationContext,
    retained: usize,
    include_bridge: bool,
) -> (
    ExponentLutNoiseSnapshot,
    PreRefreshNoiseEvaluation,
    BigUint,
    BigUint,
    mxx_exponent_lut::noise::ProgramNoiseState,
    RightActionNoiseGain,
    BigUint,
    BigUint,
) {
    use mxx_exponent_lut::{
        noise::{
            GateNoiseDecomposition, GateNoiseOperation, NoiseDecompositionPlan, ProgramNoiseState,
        },
        noise_refresh::simulation::NoiseGainPlanBuilder,
    };
    let depth = context.candidate.crt_depth;
    let ring = context.ring_dimension;
    let mut snapshot = baseline.clone();
    let mut inputs = snapshot.setup_average_inputs();
    let dropped = depth - retained;
    let params = DCRTPolyParams::new(
        ring as u32,
        depth,
        50,
        25,
        Some(context.crt_moduli.clone()),
        Some(dropped),
    );
    let epsilon = params.gadget_error_bound(None);
    let low: BigUint = context.crt_moduli[retained..].iter().map(|p| BigUint::from(*p)).product();
    // Proposition 1 residual is a sum of centered dropped CRT residues.
    // Uniform, independent residues and augmented ternary secret are the
    // explicit AverageCase assumptions. The stored convention is 4 Var.
    let residual_sum = context.crt_moduli[retained..].iter().fold(BigUint::zero(), |sum, p| {
        let p = BigUint::from(*p);
        sum + (&low / &p).pow(2) * (p.pow(2) - 1u8)
    });
    let residual = ProgramNoiseState {
        bound: &epsilon * BigUint::from(ring + 1),
        doubled_variance: AverageVariance::new(
            residual_sum * BigUint::from(3 + 2 * ring),
            9u8.into(),
        )
        .unwrap(),
    };
    if dropped != 0 {
        // Every retained balanced digit is bounded by beta/2. Retain the
        // true secret, digit, and convolution axes instead of scaling a
        // random-matrix variance without an operator bound.
        let rows = 2 * 2 * retained;
        let cap = BigUint::one() << 24usize;
        let direct = RightActionNoiseGain {
            l1: &cap * BigUint::from(rows * ring),
            l2_squared: cap.pow(2) * BigUint::from(rows * ring),
        };
        let mut gates = snapshot.gate_gains.clone();
        for gate in &mut gates {
            if gate.operation == GateNoiseOperation::Add {
                continue;
            }
            gate.action = direct.clone();
            let multiplicity = if gate.operation == GateNoiseOperation::Lut {
                gate.helper_action = Some(direct.clone());
                2u8
            } else {
                1u8
            };
            inputs.operation_residuals.insert(
                gate.operation_index,
                ProgramNoiseState {
                    bound: &residual.bound * multiplicity,
                    doubled_variance: residual
                        .doubled_variance
                        .scaled(&BigUint::from(multiplicity)),
                },
            );
        }
        snapshot.gate_gains = gates.clone();
        snapshot.readout_gains.target = direct.clone();
        snapshot.readout_gains.normalization = direct.clone();
        for trace in &mut snapshot.readout_gains.nonidentity_trace {
            trace.action = direct.clone();
            trace.helper_action = direct.clone();
        }
        let plan = NoiseDecompositionPlan::bounded_matrix(
            BigUint::one() << 25usize,
            rows,
            rows,
            ring,
            cap,
        )
        .unwrap();
        let add_plan = NoiseDecompositionPlan::unit(
            BigUint::one() << 25usize,
            2 * depth,
            2,
            4 * depth,
            ring,
            0,
            0,
            0,
        )
        .unwrap()
        .bind_crt_layout(context.crt_moduli.iter().map(|p| BigUint::from(*p)).collect(), 2)
        .unwrap();
        snapshot.gain_plan = NoiseGainPlanBuilder::new(
            gates
                .iter()
                .map(|gate| GateNoiseDecomposition {
                    action: if gate.operation == GateNoiseOperation::Add {
                        add_plan.clone()
                    } else {
                        plan.clone()
                    },
                    helper_action: gate.helper_action.as_ref().map(|_| plan.clone()),
                })
                .collect(),
            plan.clone(),
            vec![(plan.clone(), plan.clone()); snapshot.readout_gains.nonidentity_trace.len()],
            plan,
        )
        .unwrap()
        .build(gates, snapshot.readout_gains.clone())
        .unwrap();
        let zero =
            ProgramNoiseState { bound: BigUint::zero(), doubled_variance: AverageVariance::zero() };
        let trace_residual = ProgramNoiseState {
            bound: &residual.bound * (&snapshot.target_plaintext_gain().l1 + 1u8),
            doubled_variance: residual
                .doubled_variance
                .scaled(&(&snapshot.target_plaintext_gain().l2_squared + 1u8)),
        };
        // Scalar multiples of the shortened gadget have zero discarded
        // residual; the two generic trace actions do not.
        snapshot.readout_residuals = Some((
            zero.clone(),
            vec![trace_residual; snapshot.readout_gains.nonidentity_trace.len()],
            zero,
        ));
    }
    let mut pre_refresh = snapshot.evaluate_pre_refresh_with_inputs(inputs).unwrap();
    let (bridge_gain, bridge_residual) = mxx_exponent_lut::noise::even_residual_gadget_bridge(
        &context.crt_moduli,
        retained,
        &(BigUint::one() << 25usize),
        2,
    )
    .unwrap();
    let bridge_bias = &bridge_residual * 31u8 / 2u8;
    if include_bridge {
        pre_refresh.apply_gadget_bridge(&bridge_gain, &bridge_bias);
    }
    (snapshot, pre_refresh, epsilon, low, residual, bridge_gain, bridge_residual, bridge_bias)
}

fn approximate_gadget_noise_cases(
    depth: usize,
    log_ring: usize,
    mask_digits: usize,
    switch_width: usize,
    include_bridge: bool,
    retained_override: Option<&[usize]>,
) -> Vec<serde_json::Value> {
    let mut config = SearchConfig::reviewed();
    config.noise_model = NoiseSearchModel::AverageCase;
    config.sparse_lwr_universe = 8192;
    config.sparse_lwr_weight = 18;
    config.sparse_lwr_modulus = 4096;
    config.sparse_lwr_output_modulus = 32;
    assert_eq!(config.secret_dimension, 2);
    let candidate =
        Candidate { crt_depth: depth, log_ring_dimension: log_ring, crt_bits: 50, base_bits: 25 };
    let ring = 1usize << log_ring;
    let mut context = build_candidate_context(&config, candidate).unwrap();
    context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, switch_width).unwrap();
    let prepared =
        prepare_candidate_from_ring_context(&config, &context, Some((mask_digits, 1)), false)
            .unwrap();
    let baseline = prepared.noise_snapshot.unwrap();
    let run =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/profile_b_approximate_gadget_noise")
        .join(run.to_string());
    fs::create_dir_all(&directory).unwrap();
    eprintln!("APPROXIMATE_NOISE_DIRECTORY {}", directory.display());
    let mut rows = Vec::new();
    let retained_cases = retained_override
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![depth, (2 * depth).div_ceil(3), depth.div_ceil(2), depth / 2]);
    for retained in retained_cases {
        let dropped = depth - retained;
        let (
            snapshot,
            pre_refresh,
            epsilon,
            low,
            residual,
            bridge_gain,
            bridge_residual,
            bridge_bias,
        ) = modeled_approximate_pre_refresh(&baseline, &context, retained, include_bridge);
        let average = snapshot
            .simulate_average_with_pre_refresh(
                &average_case_config(&config).unwrap(),
                pre_refresh.clone(),
            )
            .unwrap();
        let program_variance = pre_refresh
            .program
            .wire_variances
            .get(snapshot.program.outputs().last().unwrap())
            .unwrap();
        let program_favg = mxx_exponent_lut::noise_refresh::simulation::average_favg(
            &average.refresh.z_squared,
            program_variance,
            0,
        )
        .unwrap();
        let readout_stochastic_favg = mxx_exponent_lut::noise_refresh::simulation::average_favg(
            &average.refresh.z_squared,
            &pre_refresh.readout_variance,
            0,
        )
        .unwrap();
        let readout_favg = &readout_stochastic_favg + &pre_refresh.readout_bias_bound;
        let row = serde_json::json!({
            "depth":depth,"ring_dimension":ring,"mask_digits":mask_digits,"switch_width":switch_width,
            "retained_towers":retained,"dropped_towers":dropped,"gadget_columns":4*retained,
            "full_modulus":snapshot.refresh.global_modulus.as_ref().unwrap().to_string(),
            "crt_moduli":context.crt_moduli,"low_modulus":low.to_string(),
            "coefficient_residual_bound":epsilon.to_string(),"secret_weighted_residual":residual,
            "program_bound":pre_refresh.program.output_bound.to_string(),
            "program_bound_bits":pre_refresh.program.output_bound.bits(),
            "readout_bound":pre_refresh.readout_bound.to_string(),"readout_bound_bits":pre_refresh.readout_bound.bits(),
            "readout_doubled_variance":pre_refresh.readout_variance,
            "bridge_included":include_bridge,"bridge_gain":bridge_gain,"bridge_residual":bridge_residual.to_string(),"bridge_bias":bridge_bias.to_string(),
            "modeled_refresh_accepted":if include_bridge {Some(average.refresh.accepted)}else{None},
            "program_F_avg":program_favg.to_string(),"program_F_avg_bits":program_favg.bits(),
            "readout_F_avg":readout_favg.to_string(),"readout_F_avg_bits":readout_favg.bits(),
            "readout_stochastic_F_avg":readout_stochastic_favg.to_string(),"readout_bias_bound":pre_refresh.readout_bias_bound.to_string(),
            "joint_event_count":average.refresh.joint_event_count.to_string(),"z_squared":average.refresh.z_squared,
            "conditional_refresh_F_avg_bits":average.refresh.max_f_avg.bits(),
            "conditional_minimum_rounding_margin":average.refresh.slots.iter().map(|s| &s.rounding_margin).min().unwrap().to_string(),
            "conditional_minimum_smudging_margin":average.refresh.slots.iter().map(|s| &s.smudging_margin).min().unwrap().to_string(),
            "baseline_first_action":baseline.gate_gains.first().map(|g|&g.action),
            "modeled_first_action":snapshot.gate_gains.first().map(|g|&g.action),
            "conditional_refresh_max_F_avg":average.refresh.max_f_avg.to_string(),
            "conditional_refresh_accepted":average.refresh.accepted,
            "full_approximate_refresh_accepted":if dropped==0 { Some(average.refresh.accepted) } else { None },
            "scope":if dropped==0 { "exact baseline" } else if include_bridge { "Approximate gadget noise model with explicit even-residual scalar bridge; not runtime or security acceptance" } else { "shortened-gadget PRF/readout model; full-width refresh figures conditional on a zero-error bridge, not executable acceptance" },
            "program_steps":pre_refresh.program.steps,
        });
        fs::write(
            directory.join(format!("retained-{retained}.json")),
            serde_json::to_vec_pretty(&row).unwrap(),
        )
        .unwrap();
        eprintln!(
            "APPROXIMATE_NOISE depth={depth} logN={log_ring} dm={mask_digits} retained={retained} columns={} program_average_bits={} readout_average_bits={} conditional_F_avg={}",
            4 * retained,
            program_favg.bits(),
            readout_favg.bits(),
            average.refresh.max_f_avg
        );
        rows.push(row);
    }
    fs::write(directory.join("summary.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "parameters":{"N":ring,"crt_depth":depth,"crt_bits":50,"base_bits":25,"security_bits":100,"v":8192,"h":18,"Q":4096,"p":32,"mask_digits":mask_digits,"fresh_digits":1,"refresh_switch_retained_towers":switch_width,"secret_dimension":2},
        "assumptions":"Uniform independent centered dropped CRT residues, augmented iid ternary secret, independent residual sources and TFHE-style propagation; stored variance is four times physical variance. See bridge_included per case: the optional custom even-residual bridge has deterministic bias propagated with l1 gains. Neither case establishes runtime or security acceptance.",
        "cases":rows,
    })).unwrap()).unwrap();
    rows
}

#[test]
#[ignore = "CPU-only inventory for an accepted real round-trip candidate"]
fn test_round_trip_artifact_inventory() {
    let config = SearchConfig::from_env().unwrap();
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT").unwrap()).unwrap(),
    )
    .unwrap();
    let result = &report["result"];
    let candidate = serde_json::from_value(result["candidate"].clone()).unwrap();
    let evidence = &result["average_evidence"];
    let (bundle, _, _) = round_trip_bundle(
        &config,
        candidate,
        evidence["mask_digit_count"].as_u64().unwrap() as usize,
        evidence["fresh_error_digit_count"].as_u64().unwrap() as usize,
        1,
    )
    .unwrap();
    for manifest in bundle.preprocessing_manifests().values() {
        println!("artifact_inventory={}", serde_json::to_string(manifest).unwrap());
    }
}

fn reviewed_estimator_bits(universe: usize, q_l: usize) -> u64 {
    match (universe, q_l) {
        (16384, 2048) => 100,
        (16384, 4096) => 100,
        (8192, 4096) => 103,
        other => panic!("unexpected reviewed tuple {other:?}"),
    }
}

fn mock_prepared(candidate: Candidate) -> PreparedCandidate {
    PreparedCandidate {
        anchor_security_inputs: Vec::new(),
        candidate,
        ring_dimension: 1 << candidate.log_ring_dimension,
        bucket_width: 3,
        official_preimage_bound: 10.into(),
        selected_preimage_bound: 10.into(),
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
        noise_snapshot: None,
        pre_refresh: None,
        production_report: None,
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
        tier: SparseLwrSecurityTier::Requested,
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
    assert_eq!(config.crt_depths, 1..=62);
    assert_eq!(config.log_ring_dimensions, 16..=16);
    assert_eq!(config.crt_bits, 50);
    assert_eq!(config.base_bits, 25);
    assert_eq!(config.crt_base_bits_grid, vec![(50, 25)]);
    assert_eq!(config.crt_base_bits_grid[0], (50, 25));
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
            Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
            Candidate { crt_depth: 1, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
            Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
            Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 30, base_bits: 15 },
            Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 15 },
        ]
    );
}

#[test]
fn average_policy_is_derived_once_from_search_config() {
    let mut config = SearchConfig::reviewed();
    config.security_bits = 37;
    let policy = average_case_config(&config).unwrap();
    assert_eq!(policy.failure_exponent, 37);
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
fn phase_two_does_not_apply_a_hidden_total_modulus_bit_cap() {
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
    assert_eq!(
        candidates(&config).collect::<Vec<_>>(),
        vec![Candidate { crt_depth: 63, log_ring_dimension: 16, crt_bits: 32, base_bits: 16 }]
    );
}

#[test]
fn lookup_overlay_applies_the_configured_tiny_profile() {
    let values = std::collections::BTreeMap::from([
        ("MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS", "1"),
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE", "4"),
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", "1"),
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", "4"),
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", "2"),
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
    assert_eq!(config.sparse_lwr_phase1_grid.len(), 1);
    assert_eq!(
        (
            &config.sparse_lwr_phase1_grid[0].nu,
            &config.sparse_lwr_phase1_grid[0].h,
            &config.sparse_lwr_phase1_grid[0].q_l,
            &config.sparse_lwr_phase1_grid[0].p
        ),
        (&4, &1, &4, &2)
    );
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
    assert_eq!(config.sparse_lwr_universe_grid, SearchConfig::reviewed().sparse_lwr_universe_grid);
}

#[test]
fn lookup_overlay_preserves_an_explicit_crt_base_singleton() {
    let values = std::collections::BTreeMap::from([
        ("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", "50"),
        ("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", "25"),
    ]);
    let config =
        SearchConfig::from_lookup(|name| Ok(values.get(name).map(|value| (*value).to_owned())))
            .unwrap();
    assert_eq!(config.crt_bits, 50);
    assert_eq!(config.base_bits, 25);
    assert_eq!(config.crt_base_bits_grid, vec![(50, 25)]);
    let mut bounded = config.clone();
    bounded.crt_depths = 1..=2;
    bounded.log_ring_dimensions = 16..=16;
    assert!(
        candidates(&bounded)
            .all(|candidate| { candidate.crt_bits == 50 && candidate.base_bits == 25 })
    );
}

#[test]
fn lookup_overlay_accepts_reviewed_log_ring_dimension_and_crt_base_pair() {
    let values = std::collections::BTreeMap::from([
        ("MXX_EXPONENT_LUT_REFRESH_MIN_LOG_RING_DIMENSION", "16"),
        ("MXX_EXPONENT_LUT_REFRESH_MAX_LOG_RING_DIMENSION", "16"),
        ("MXX_EXPONENT_LUT_REFRESH_CRT_BITS", "50"),
        ("MXX_EXPONENT_LUT_REFRESH_BASE_BITS", "25"),
    ]);
    let config =
        SearchConfig::from_lookup(|name| Ok(values.get(name).map(|value| (*value).to_owned())))
            .unwrap();
    assert_eq!(config.log_ring_dimensions, 16..=16);
    assert_eq!(config.crt_base_bits_grid, vec![(50, 25)]);
}

#[test]
fn backend_capability_accepts_the_exact_50_25_singleton() {
    assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 50, 25).is_ok());
    assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 60, 30).is_ok());
    assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 60, 31).is_err());
    assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 61, 25).is_err());
    assert!(DCRTPolyParams::validate_capability(1 << 16, 1, 50, 32).is_err());
}

#[test]
fn reviewed_bgg_envelope_accepts_only_its_bound_domain() {
    let mut config = SearchConfig::reviewed();
    config.bgg_security_envelope = Some("reviewed_monotone_envelope".to_owned());
    let accepted = Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
    assert_eq!(reviewed_bgg_security_bits(&config, accepted), Ok(136));
    assert_eq!(bgg_security_authority(&config), "reviewed_monotone_envelope");
    assert_eq!(bgg_security_evidence_source(&config), "reviewed_monotone_envelope:q27:136");

    for candidate in [
        Candidate { crt_depth: 28, ..accepted },
        Candidate { log_ring_dimension: 15, ..accepted },
        Candidate { crt_bits: 49, ..accepted },
        Candidate { base_bits: 24, ..accepted },
    ] {
        assert!(reviewed_bgg_security_bits(&config, candidate).is_err());
    }

    let mut wrong_sigma = config.clone();
    wrong_sigma.error_sigma = 4.000000000000001;
    assert!(reviewed_bgg_security_bits(&wrong_sigma, accepted).is_err());
}

#[test]
fn reviewed_bgg_envelope_requires_explicit_selection() {
    let config = SearchConfig::reviewed();
    let candidate =
        Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
    assert!(reviewed_bgg_security_bits(&config, candidate).is_err());
    let error = SearchConfig::from_lookup(|name| {
        Ok((name == "MXX_EXPONENT_LUT_REFRESH_BGG_SECURITY_ENVELOPE")
            .then(|| "unreviewed".to_owned()))
    })
    .unwrap_err();
    assert!(error.contains("reviewed_monotone_envelope"));
}

#[test]
fn explicit_reviewed_tuple_reuses_pinned_estimator_evidence() {
    let values = std::collections::BTreeMap::from([
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE", "16384"),
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_WEIGHT", "17"),
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_MODULUS", "2048"),
        ("MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_OUTPUT_MODULUS", "16"),
    ]);
    let config =
        SearchConfig::from_lookup(|name| Ok(values.get(name).map(|value| (*value).to_owned())))
            .unwrap();
    let tuple = &config.sparse_lwr_phase1_grid[0];
    assert_eq!(tuple.estimator_security_bits, Some(100));
    assert_eq!(tuple.estimator_minimum_classical_bits, Some(100.5636531151896));
}

#[test]
fn decoder_cutoff_acceptance_is_monotone_from_the_official_bound_in_setup() {
    let baseline = RefreshSetupParameters::new(
        [0x91; 32],
        2,
        2,
        4,
        1,
        1,
        32,
        4,
        BggSamplerLayout {
            modulus: 6.into(),
            ring_dimension: 4.into(),
            secret_dimension: 2,
            digit_count: 2,
            gadget_base: 2.into(),
        },
        RefreshCompiler {
            full_modulus: 6.into(),
            crt_plaintext_moduli: vec![2.into(), 3.into()],
            reconstruction_coefficients: vec![1.into(), 1.into()],
        },
        1.into(),
        1.into(),
        "cutoff-boundary-test",
    );
    let official = baseline
        .resolve_decoder_preimage_bound()
        .unwrap()
        .evaluate(&mxx_ir_core::ParamEnv::default())
        .unwrap();
    let official = official.to_biguint().unwrap();
    for (offset, accepted) in [(-1_i32, false), (0, true), (1, true)] {
        let selected = if offset < 0 {
            &official - BigUint::from((-offset) as u32)
        } else {
            &official + BigUint::from(offset as u32)
        };
        assert_eq!(validate_decoder_preimage_cutoff(&selected, &official).is_ok(), accepted);
        if offset < 0 {
            let policy = TrapdoorPreimageCutoffPolicy::Explicit(selected);
            assert!(
                baseline
                    .clone()
                    .with_decoder_preimage_bound(policy)
                    .resolve_decoder_preimage_bound()
                    .is_err()
            );
            continue;
        }
        let policy = TrapdoorPreimageCutoffPolicy::Explicit(selected.clone());
        let resolved = baseline
            .clone()
            .with_decoder_preimage_bound(policy)
            .resolve_decoder_preimage_bound()
            .unwrap()
            .evaluate(&mxx_ir_core::ParamEnv::default())
            .unwrap()
            .to_biguint()
            .unwrap();
        assert_eq!(resolved, selected);
        if offset == 1 {
            // This is an actual setup clone carrying Explicit(Default+1),
            // rather than a detached comparison of two integers.
            assert!(resolved > official);
        }
    }
}

#[test]
fn lookup_overlay_rejects_a_partial_crt_base_pair() {
    let error = SearchConfig::from_lookup(|name| {
        Ok((name == "MXX_EXPONENT_LUT_REFRESH_CRT_BITS").then(|| "50".to_owned()))
    })
    .unwrap_err();
    assert!(error.contains("supplied together"));
}

#[test]
fn lookup_overlay_rejects_a_partial_native_phase_one_tuple() {
    let error = SearchConfig::from_lookup(|name| {
        Ok((name == "MXX_EXPONENT_LUT_REFRESH_SPARSE_LWR_UNIVERSE").then(|| "16384".to_owned()))
    })
    .unwrap_err();
    assert!(error.contains("UNIVERSE, _WEIGHT, _MODULUS, and _OUTPUT_MODULUS"));
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
fn requested_one_bit_target_controls_both_phases_and_checkpoint_identity() {
    let mut config = SearchConfig::reviewed();
    config.security_bits = 1;
    config.mask_statistical_security_bits = 1;
    config.sparse_lwr_universe = 64;
    config.sparse_lwr_weight = 1;
    config.sparse_lwr_modulus = 8;
    config.sparse_lwr_output_modulus = 4;
    config.sparse_lwr_universe_grid = vec![64];
    config.sparse_lwr_phase1_grid = vec![SparseLwrParameterTuple {
        q_l: 8,
        p: 4,
        nu: 64,
        h: 1,
        estimator_minimum_classical_bits: None,
        estimator_security_bits: None,
    }];
    config.crt_depths = 1..=1;
    config.log_ring_dimensions = 5..=5;
    let selected = select_sparse_lwr_profile(&config, |_, _, _, _| Ok(16)).unwrap();
    assert_eq!(selected.tier, SparseLwrSecurityTier::Requested);
    let checkpoint = Phase1Checkpoint::from_selection(&config, &selected).unwrap();
    checkpoint.validate(&config).unwrap();
    let mut different_target = config.clone();
    different_target.security_bits = 2;
    assert!(checkpoint.validate(&different_target).is_err());
    let result =
        search_with_hooks(&config, &selected, |c| Ok(mock_prepared(c)), |_| Ok(1), |_| Ok(true))
            .unwrap();
    assert_eq!(result.achieved_security_bits, 1);
    let report = search_report(&config, &selected, result);
    assert_eq!(report.phase1.security_target_bits, 1);
    assert_eq!(report.phase2.security_target_bits, 1);
    assert!(
        search_with_hooks(&config, &selected, |c| Ok(mock_prepared(c)), |_| Ok(0), |_| Ok(true))
            .is_err()
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
    assert_eq!(calls, vec![16384, 16384, 8192]);
    assert_eq!(selected.universe, 16384);
    assert_eq!(selected.q_l, 2048);
    assert_eq!(selected.tier, SparseLwrSecurityTier::Requested);
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
    assert_eq!(
        profiles
            .iter()
            .map(|profile| (profile.q_l, profile.p, profile.universe))
            .collect::<Vec<_>>(),
        [(2048, 16, 16384), (4096, 32, 16384), (4096, 32, 8192)]
    );
    assert!(profiles.iter().all(|profile| profile.tier == SparseLwrSecurityTier::Requested));
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
    assert!(error.contains("tuple grid"));
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
    assert_eq!(second.universe, 16384);
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
    assert_eq!(selected.universe, 16384);

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
    // Keep the tampered tuple structurally valid so this exercises the
    // checkpoint's source-of-truth identity check rather than native-Q
    // validation itself.
    changed.sparse_lwr_phase1_grid[0].q_l = 1024;
    assert!(
        load_or_search_phase1(&changed, Some(&path), |universe, _, q_l, _| Ok(
            reviewed_estimator_bits(universe, q_l)
        ))
        .unwrap_err()
        .contains("does not match")
    );
    let mut checkpoint: Phase1Checkpoint =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    checkpoint.selected.tier = SparseLwrSecurityTier::Rejected;
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
    changed.security_bits = 101;
    assert!(
        load_or_search_phase1(&changed, Some(&path), |_, _, _, _| Ok(128))
            .unwrap_err()
            .contains("does not match")
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn phase_one_checkpoint_round_trips_the_boundary_tuple() {
    for selected_universe in [16384, 8192] {
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
fn accepted_phase_two_profile_persists_native_q_and_rejects_out_of_grid_tampering() {
    let config = SearchConfig::reviewed();
    let path = checkpoint_test_path("phase2-accepted");
    let _ = std::fs::remove_file(&path);
    let selected = load_or_search_phase1(&config, Some(&path), |universe, _, q_l, _| {
        Ok(reviewed_estimator_bits(universe, q_l))
    })
    .unwrap();
    let mut order = Vec::new();
    let (selected_profile, result) = search_qualified_profiles(&selected, |profile| {
        order.push(profile.q_l);
        Ok(SearchResult {
            candidate: Candidate {
                crt_depth: match profile.q_l {
                    2048 => 30,
                    4096 if profile.universe == 16384 => 28,
                    4096 => 29,
                    _ => 31,
                },
                log_ring_dimension: 16,
                crt_bits: 50,
                base_bits: 25,
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
            switched_security_status: "low-P marginal estimates not supplied; cross-modulus joint-view security unresolved".into(),
            sparse_lwr_q_l: profile.q_l,
            sparse_lwr_p: profile.p,
            sparse_lwr_tier: profile.tier,
            estimator_commit: profile.estimator_commit.clone(),
            estimator_cost_model: profile.estimator_cost_model.clone(),
            estimator_shape_model: profile.estimator_shape_model.clone(),
            average_evidence: None,
            production_report: None,
            bgg_security_authority: "external_lattice_estimator".to_owned(),
            bgg_security_evidence_source: "external_lattice_estimator".to_owned(),
        })
    })
    .unwrap();
    // Every reviewed profile is evaluated, and global selection chooses the
    // later profile with the smallest CRT depth.
    assert_eq!(order, vec![2048, 4096, 4096]);
    assert_eq!(selected_profile.q_l, 4096);
    assert_eq!(selected_profile.universe, 16384);
    assert_eq!(result.candidate.crt_depth, 28);
    let candidate = result.candidate;
    persist_accepted_phase2_profile(&path, &config, &selected_profile, candidate).unwrap();
    let checkpoint: Phase1Checkpoint =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    checkpoint.validate(&config).unwrap();
    // The Phase-2 global winner is intentionally allowed to differ from the
    // Phase-1 representative stored in `checkpoint.selected`.
    assert_ne!(checkpoint.selected.tuple, selected_profile.tuple);
    assert_eq!(checkpoint.accepted_phase2.as_ref().unwrap().tuple.q_l, 4096);
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
    let error =
        search_qualified_profiles(&selected, |_| Err("fatal setup error".to_owned())).unwrap_err();
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
        candidate: Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
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
        switched_security_status:
            "low-P marginal estimates not supplied; cross-modulus joint-view security unresolved"
                .into(),
        sparse_lwr_q_l: 8,
        sparse_lwr_p: 2,
        sparse_lwr_tier: SparseLwrSecurityTier::Requested,
        estimator_commit: REVIEWED_ESTIMATOR_COMMIT.to_owned(),
        estimator_cost_model: REVIEWED_ESTIMATOR_COST_MODEL.to_owned(),
        estimator_shape_model: REVIEWED_ESTIMATOR_SHAPE_MODEL.to_owned(),
        average_evidence: None,
        production_report: None,
        bgg_security_authority: "external_lattice_estimator".to_owned(),
        bgg_security_evidence_source: "external_lattice_estimator".to_owned(),
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
        candidate: Candidate { crt_depth: 2, log_ring_dimension: 5, crt_bits: 32, base_bits: 16 },
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
        switched_security_status:
            "low-P marginal estimates not supplied; cross-modulus joint-view security unresolved"
                .into(),
        sparse_lwr_q_l: sparse_profile.q_l,
        sparse_lwr_p: sparse_profile.p,
        sparse_lwr_tier: sparse_profile.tier,
        estimator_commit: sparse_profile.estimator_commit.clone(),
        estimator_cost_model: sparse_profile.estimator_cost_model.clone(),
        estimator_shape_model: sparse_profile.estimator_shape_model.clone(),
        average_evidence: None,
        production_report: None,
        bgg_security_authority: "external_lattice_estimator".to_owned(),
        bgg_security_evidence_source: "external_lattice_estimator".to_owned(),
    };
    let json = serde_json::to_string(&search_report(&config, &sparse_profile, result)).unwrap();

    assert!(json.contains("\"parameter_grid\":["));
    assert!(json.contains("\"universe_grid\":[8192,16384]"));
    assert!(json.contains(&format!("\"support_weight\":{}", sparse_profile.weight)));
    assert!(json.contains("\"q_l\":2048"));
    assert!(json.contains("\"output_modulus\":16"));
    assert!(json.contains("\"security_target_bits\":100"));
    assert!(json.contains("\"sparse_secret_model\":\"SparseBinary\""));
    assert!(json.contains("\"sparse_error_model\":\"Uniform\""));
    assert!(json.contains("\"selected_error_lower\":-64"));
    assert!(json.contains("\"selected_error_upper\":63"));
    assert!(json.contains("\"exact_estimator\":true"));
    assert!(json.contains("\"evaluated\":["));
    assert!(json.contains("\"universe\":16384"));
    assert!(json.contains("\"minimum_security_bits\""));
    assert!(json.contains("\"qualified\":true"));
    assert!(json.contains("\"selected_universe\":16384"));
    assert!(json.contains("\"crt_depth_min\":2"));
    assert!(json.contains("\"crt_depth_max\":3"));
    assert!(json.contains("\"log_ring_dimension_min\":5"));
    assert!(json.contains("\"log_ring_dimension_max\":6"));
    assert!(
        json.contains("\"order\":\"crt_depth_ascending_then_log_ring_dimension_ascending_then_base_bits_descending_then_crt_bits_descending; profile_tuple_q_l_p_nu_h_ascending\"")
        );
    assert!(json.contains("\"crt_base_bits_grid\":[[50,25]"));
    assert!(!json.contains("selected_slots"));
    assert!(!json.contains("support_coordinates"));
}

#[test]
fn sparse_lwr_estimator_arguments_use_exact_reviewed_model() {
    assert_eq!(sparse_lwr_error_bounds(2048, 16).unwrap(), (-64, 63));
    assert_eq!(
        sparse_lwr_estimator_args(16384, 17, 2048, 16).unwrap(),
        vec![
            "16384",
            "2048",
            "--s-dist",
            r#"{"name":"SparseBinary","hw":17,"n":16384}"#,
            "--e-dist",
            r#"{"name":"Uniform","a":-64,"b":63}"#,
            "--exact",
        ]
    );
    assert!(sparse_lwr_error_bounds(15, 2).is_err());
    assert!(sparse_lwr_error_bounds(2048, 0).is_err());
}

#[test]
fn reconstruction_coefficients_come_from_the_dcrt_parameters() {
    let dcrt = DCRTPolyParams::new(32, 2, 28, 14, None, None);
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
    assert_eq!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[37, 41]), Ok(1));
    assert!(fresh_error_base_p_digit_count(&BigUint::from(32u8), &[31, 32]).is_err());
}

#[test]
fn fresh_digit_selection_checks_prime_factors_of_composite_towers() {
    // B_e = 7 is below the composite modulus 35, but not below its prime
    // factor 5; the modulus-level comparison must therefore reject it.
    assert!(fresh_error_base_p_digit_count(&BigUint::from(8u8), &[35]).is_err());
    assert_eq!(fresh_error_base_p_digit_count(&BigUint::from(4u8), &[35]), Ok(1));
}

#[test]
fn coarse_switch_ring_candidates_preserve_slots_and_evaluate_real_noise() {
    assert_eq!(
        candidate_switch_moduli(&[3, 5, 7], 2).unwrap(),
        vec![15u8.into(), 15u8.into(), 35u8.into()]
    );
    assert_eq!(candidate_switch_moduli(&[3, 5, 7], 3).unwrap(), vec![105u8.into(); 3]);
    assert!(candidate_switch_moduli(&[3, 5, 7], 0).is_err());
    assert!(
        mask_digit_search_bounds(
            &4u8.into(),
            &4u8.into(),
            &[3, 5, 7],
            &candidate_switch_moduli(&[3, 5, 7], 1).unwrap()
        )
        .is_err()
    );
    let mut config = SearchConfig::reviewed();
    config.sparse_lwr_universe = 10;
    config.sparse_lwr_weight = 4;
    config.sparse_lwr_modulus = 32;
    config.sparse_lwr_output_modulus = 4;
    let candidate = Candidate { crt_depth: 3, log_ring_dimension: 5, crt_bits: 40, base_bits: 4 };
    match prepare_candidate(&config, candidate) {
        Ok(mut prepared) => {
            assert!(check_refresh_bundle(&config, candidate, &mut prepared).unwrap());
            let snapshot = prepared.noise_snapshot.as_ref().unwrap();
            assert_eq!(snapshot.refresh.switch_moduli.len(), 3);
            eprintln!(
                "coarse switched candidate accepted by configured noise authority: P={:?}, report={}",
                snapshot.refresh.switch_moduli,
                serde_json::to_string(prepared.production_report.as_ref().unwrap()).unwrap()
            );
        }
        Err(CandidatePreparationError::Infeasible(reason)) => {
            assert!(reason.contains("coarse switched-ring scan rejected"));
            assert!((1..=3).all(|retained| reason.contains(&format!("retained={retained}:"))));
            eprintln!("coarse switched candidate concrete rejection: {reason}");
        }
        Err(error) => panic!("unexpected candidate construction failure: {error:?}"),
    }
}

#[test]
fn profile_b_switched_refresh_named_average_authority_component_search() {
    // Pure public-layout/noise interpretation: no backend, artifact sampling,
    // external estimator, integration execution, or benchmark is invoked.
    let mut config = SearchConfig::reviewed();
    config.noise_model = NoiseSearchModel::AverageCase;
    config.crt_depths = 27..=27;
    config.log_ring_dimensions = 16..=16;
    config.sparse_lwr_universe = 8192;
    config.sparse_lwr_weight = 18;
    config.sparse_lwr_modulus = 4096;
    config.sparse_lwr_output_modulus = 32;
    let candidate =
        Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
    eprintln!(
        "Profile B: n=65536, depth=27, crt_bits=50, base_bits=25, Q=4096, p=32, v=8192, h=18; security target=100; authority={:?}; coarse contiguous retained-width groups and uniform mask digits; security acceptance is NOT inferred from noise acceptance",
        average_case_config(&config).unwrap()
    );
    match prepare_candidate(&config, candidate) {
        Ok(mut prepared) => {
            assert!(check_refresh_bundle(&config, candidate, &mut prepared).unwrap());
            let report = prepared.production_report.as_ref().unwrap();
            assert!(report.average_case.as_ref().unwrap().accepted);
            assert_eq!(report.anchor_security_inputs.len(), 6);
            assert!(report.anchor_security_inputs.iter().all(
                |group| group.modulus.bits() == 250 &&
                    group.ring_dimension == 65536 &&
                    group.anchor_columns == 24 &&
                    group.execution_budget == 1 &&
                    group.estimated_security_bits.is_none()
            ));
            assert_eq!(prepared.mask_base_p_digit_count, 38);
            eprintln!("PROFILE_B_NOISE_ACCEPTED {}", serde_json::to_string(report).unwrap());
        }
        Err(CandidatePreparationError::Infeasible(reason)) => {
            panic!("previously accepted Profile B noise candidate regressed: {reason}");
        }
        Err(error) => panic!("Profile B component construction failed: {error:?}"),
    }
}

#[test]
#[ignore = "explicit full retained-width parameter search; no backend execution"]
fn profile_b_minimum_mask_retained_width_search() {
    let mut config = SearchConfig::reviewed();
    config.noise_model = NoiseSearchModel::AverageCase;
    config.crt_depths = 27..=27;
    config.log_ring_dimensions = 16..=16;
    config.sparse_lwr_universe = 8192;
    config.sparse_lwr_weight = 18;
    config.sparse_lwr_modulus = 4096;
    config.sparse_lwr_output_modulus = 32;
    let candidate =
        Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
    let mut context = build_candidate_context(&config, candidate).unwrap();
    let started = std::time::Instant::now();
    let run =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/profile_b_minimum_mask_search")
        .join(run.to_string());
    fs::create_dir_all(&directory).unwrap();
    eprintln!("SEARCH_DIRECTORY {}", directory.display());
    let mut rows = Vec::new();
    let mut selected: Option<(usize, usize)> = None;
    for retained in 1..=candidate.crt_depth {
        eprintln!(
            "SEARCH_WIDTH_START retained={retained} elapsed={:.3}s",
            started.elapsed().as_secs_f64()
        );
        context.switch_moduli = candidate_switch_moduli(&context.crt_moduli, retained).unwrap();
        match prepare_candidate_from_ring_context(&config, &context, None, false) {
            Ok(mut prepared) => {
                assert!(check_refresh_bundle(&config, candidate, &mut prepared).unwrap());
                let report = prepared.production_report.as_ref().unwrap();
                let average = report.average_case.as_ref().unwrap();
                let snapshot = prepared.noise_snapshot.as_ref().unwrap();
                let mask_digits = prepared.mask_base_p_digit_count;
                let score = (candidate.crt_depth * mask_digits, retained);
                if selected.is_none_or(|best| score < best) {
                    selected = Some(score);
                }
                let row = serde_json::json!({
                    "retained": retained, "accepted": true,
                    "mask_digits": mask_digits, "sum_mask_digits": score.0,
                    "fresh_digits": prepared.fresh_error_base_p_digit_count,
                    "label_count": prepared.prf_label_count,
                    "q": snapshot.refresh.global_modulus.as_ref().unwrap().to_string(),
                    "q_t": snapshot.refresh.plaintext_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "P_t": snapshot.refresh.switch_moduli.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "M": snapshot.refresh.base_p.pow(mask_digits as u32).to_string(),
                    "max_F_avg": average.refresh.max_f_avg.to_string(),
                    "minimum_rounding_margin": average.refresh.slots.iter().map(|slot| &slot.rounding_margin).min().unwrap().to_string(),
                    "minimum_smudging_margin": average.refresh.slots.iter().map(|slot| &slot.smudging_margin).min().unwrap().to_string(),
                    "joint_event_count": average.refresh.joint_event_count.to_string(),
                    "z_squared": format!("{}/{}", average.refresh.z_squared.numerator, average.refresh.z_squared.denominator),
                    "anchor_group_count": report.anchor_security_inputs.len(),
                    "decoder_columns": report.anchor_security_inputs.iter().map(|group| group.anchor_columns).collect::<Vec<_>>(),
                    "official_preimage_bound": prepared.official_preimage_bound.to_string(),
                    "selected_preimage_bound": prepared.selected_preimage_bound.to_string(),
                    "fresh_margins": report.fresh_error_margins,
                    "slots": average.refresh.slots.iter().map(|slot| serde_json::json!({
                        "slot": slot.slot, "P": slot.switch_modulus.to_string(), "J": slot.lift.to_string(),
                        "Delta": slot.spacing.to_string(), "F_avg": slot.f_avg.to_string(),
                        "B_m": slot.deterministic_mask_bound.to_string(),
                        "rounding_margin": slot.rounding_margin.to_string(),
                        "smudging_margin": slot.smudging_margin.to_string(),
                    })).collect::<Vec<_>>(),
                });
                fs::write(
                    directory.join(format!("retained-{retained}-full-report.json")),
                    serde_json::to_vec_pretty(report).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "SEARCH_WIDTH_ACCEPT retained={retained} dm={mask_digits} labels={} elapsed={:.3}s",
                    prepared.prf_label_count,
                    started.elapsed().as_secs_f64()
                );
                rows.push(row);
            }
            Err(CandidatePreparationError::Infeasible(reason)) => {
                eprintln!(
                    "SEARCH_WIDTH_REJECT retained={retained} reason={reason} elapsed={:.3}s",
                    started.elapsed().as_secs_f64()
                );
                rows.push(
                    serde_json::json!({"retained": retained, "accepted": false, "reason": reason}),
                );
            }
            Err(error) => panic!("unexpected search error at retained={retained}: {error:?}"),
        }
        fs::write(directory.join("widths.json"), serde_json::to_vec_pretty(&rows).unwrap())
            .unwrap();
    }
    let (sum_digits, retained) = selected.expect("no accepted candidate in the declared grid");
    let summary = serde_json::json!({
        "scope": "minimum sum of mask digits in all retained widths 1..27, contiguous groups, uniform digits; not exhaustive subsets or ragged optimum",
        "baseline": {"n":65536,"crt_depth":27,"crt_bits":50,"base_bits":25,"Q":4096,"p":32,"v":8192,"h":18,"encoding_sigma":4.0,"decoder_sigma":4.578},
        "noise_authority":"AverageCase TFHE-style heuristic with conservative Add",
        "failure_exponent":100,"mask_statistical_security_bits":100,"tail_correction_bits":0,
        "selected_retained":retained,"sum_mask_digits":sum_digits,
        "joint_security":"unresolved; noise acceptance is not a whole-construction security claim",
        "elapsed_seconds":started.elapsed().as_secs_f64(), "candidates":rows,
    });
    fs::write(directory.join("search.json"), serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
    eprintln!(
        "SEARCH_COMPLETE retained={retained} sum_mask_digits={sum_digits} report={}",
        directory.join("search.json").display()
    );
}

#[test]
#[ignore = "export an already accepted Profile B candidate; no simulation or backend execution"]
fn export_profile_b_modswitch_benchmark_inputs() {
    let mut config = SearchConfig::from_env().unwrap();
    assert_eq!(config.noise_model, NoiseSearchModel::AverageCase);
    assert_eq!(
        (
            config.sparse_lwr_modulus,
            config.sparse_lwr_output_modulus,
            config.sparse_lwr_universe,
            config.sparse_lwr_weight
        ),
        (4096, 32, 8192, 18)
    );
    assert_eq!((config.security_bits, config.mask_statistical_security_bits), (100, 100));
    config.bgg_security_envelope = Some("reviewed_monotone_envelope".into());
    let candidate =
        Candidate { crt_depth: 27, log_ring_dimension: 16, crt_bits: 50, base_bits: 25 };
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data/profile_b_minimum_mask_search/1788588283213876339");
    let saved: ProductionNoiseReport =
        serde_json::from_slice(&fs::read(directory.join("retained-5-full-report.json")).unwrap())
            .unwrap();
    let average = saved.average_case.as_ref().expect("saved average authority");
    assert!(average.accepted && average.hard_authority_accepted && average.correctness_accepted);
    let selected: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.join("search.json")).unwrap()).unwrap();
    assert_eq!(selected["selected_retained"].as_u64(), Some(5));
    let row = selected["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["retained"].as_u64() == Some(5))
        .unwrap();
    let mask_digits = row["mask_digits"].as_u64().unwrap() as usize;
    let fresh_digits = row["fresh_digits"].as_u64().unwrap() as usize;
    let moduli = row["P_t"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap().parse::<BigUint>().unwrap())
        .collect::<Vec<_>>();
    let mut context = build_candidate_context(&config, candidate).unwrap();
    context.switch_moduli = moduli.clone();
    let setup = setup_for_candidate_context(&config, &context, mask_digits, fresh_digits).unwrap();
    // Reconstruct actual public identities and route metadata, without
    // evaluate_pre_refresh/simulate/check_refresh_bundle or an estimator.
    let snapshot = RefreshParameterSimulationRequest::build_noise_snapshot_from_public_layout(
        setup,
        context.profile,
        &context.generated_layout.public_layout,
        &context.active_cells,
    )
    .unwrap();
    assert_eq!(snapshot.setup_identity(), &average.snapshot_identity);
    assert_eq!(snapshot.refresh.switch_moduli, moduli);
    let mut prepared =
        prepare_candidate_with_fixed_digits(&config, candidate, mask_digits, fresh_digits, &moduli)
            .unwrap();
    assert_eq!(prepared.bundle.as_ref().unwrap().public_identity(), &average.snapshot_identity);
    assert_eq!(prepared.anchor_security_inputs, saved.anchor_security_inputs);
    assert_eq!(prepared.prf_label_count as u64, row["label_count"].as_u64().unwrap());
    prepared.noise_snapshot = Some(snapshot);
    let phase1 = select_sparse_lwr_profile(&config, |nu, h, q, p| {
        let tuple = config
            .sparse_lwr_phase1_grid
            .iter()
            .find(|tuple| (tuple.nu, tuple.h, tuple.q_l, tuple.p) == (nu, h, q, p))
            .ok_or("tuple is not declared")?;
        sparse_lwr_parameter_support::pinned_estimator_security_bits(tuple)?
            .ok_or_else(|| "missing pinned sparse-LWR evidence".into())
    })
    .unwrap();
    let bgg = reviewed_bgg_security_bits(&config, candidate).unwrap();
    let result = SearchResult {
        candidate,
        achieved_security_bits: bgg.min(phase1.sparse_lwr_security_bits),
        switched_security_status: "selected low-P marginal evidence is external; cross-modulus joint-view security unresolved".into(),
        bgg_rlwe_security_bits: bgg,
        sparse_lwr_security_bits: phase1.sparse_lwr_security_bits,
        raw_key_entropy_bits: phase1.raw_key_entropy_bits,
        sparse_lwr_universe: phase1.universe,
        sparse_lwr_weight: phase1.weight,
        official_preimage_bound: prepared.official_preimage_bound.to_string(),
        ring_dimension: prepared.ring_dimension,
        bucket_width: prepared.bucket_width,
        pbc_attempts_used: prepared.pbc_attempts_used,
        layout_id: hex(prepared.layout_id),
        program_id: hex(prepared.program_id),
        checker_accepted: average.accepted,
        sparse_lwr_q_l: phase1.q_l,
        sparse_lwr_p: phase1.p,
        sparse_lwr_tier: phase1.tier,
        estimator_commit: phase1.estimator_commit.clone(),
        estimator_cost_model: phase1.estimator_cost_model.clone(),
        estimator_shape_model: phase1.estimator_shape_model.clone(),
        average_evidence: final_average_evidence(&config, &prepared, average).unwrap(),
        production_report: Some(saved.clone()),
        bgg_security_authority: bgg_security_authority(&config).into(),
        bgg_security_evidence_source: bgg_security_evidence_source(&config).into(),
    };
    let mut checkpoint = Phase1Checkpoint::from_selection(&config, &phase1).unwrap();
    checkpoint.accepted_phase2 =
        Some(AcceptedPhase2Profile { tuple: phase1.tuple.clone(), candidate });
    checkpoint.validate(&config).unwrap();
    let checkpoint_path = std::env::var(PHASE1_CHECKPOINT_ENV).expect("output checkpoint path");
    let report_path =
        std::env::var("MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT").expect("output report path");
    let report = search_report(&config, &phase1, result);
    for (path, value) in [
        (checkpoint_path, serde_json::to_value(&checkpoint).unwrap()),
        (report_path, serde_json::to_value(&report).unwrap()),
    ] {
        let output = fs::OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        serde_json::to_writer_pretty(output, &value).unwrap();
        eprintln!("EXPORTED {path}");
    }
    eprintln!(
        "EXPORT_COMPLETE layout={} program={} setup={} dm={mask_digits} de={fresh_digits} labels={} simulation=false lattice_estimator=false backend_execution=false",
        hex(prepared.layout_id),
        hex(prepared.program_id),
        hex(average.snapshot_identity),
        prepared.prf_label_count
    );
}

#[test]
fn mask_digit_search_starts_at_domain_coverage_and_stops_at_safe_bound() {
    assert_eq!(
        mask_digit_search_bounds(
            &BigUint::from(2u8),
            &BigUint::from(8u8),
            &[100, 200],
            &[20000u32.into(), 20000u32.into()]
        ),
        Ok((3, 6))
    );
    assert_eq!(
        mask_digit_search_bounds(
            &BigUint::from(32u8),
            &BigUint::from(32u8),
            &[32, 37],
            &[1184u32.into(), 1184u32.into()]
        ),
        Err(CandidatePreparationError::Infeasible(
            "strict deterministic mask boundary is below the domain minimum".into()
        ))
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
    let dcrt = DCRTPolyParams::new(ring_dimension as u32, 62, 32, 1, None, None);
    let q_bits = dcrt.modulus().bits();
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
    match result {
        Ok(mut prepared) => {
            let checked = check_refresh_bundle(&config, prepared.candidate, &mut prepared)
                .expect("compact noise checker must complete");
            assert!(checked, "reviewed fixed candidate must pass canonical compact acceptance");
        }
        Err(CandidatePreparationError::Infeasible(reason)) => {
            assert!(reason.starts_with("coarse switched-ring scan rejected: retained=1:"));
            assert!(reason.contains("no mask digit count passed the complete production report"));
            assert!((1..=30).all(|retained| reason.contains(&format!("retained={retained}:"))));
        }
        Err(CandidatePreparationError::Fatal(reason)) => {
            panic!("compact preparation failed fatally: {reason}")
        }
    }
}
