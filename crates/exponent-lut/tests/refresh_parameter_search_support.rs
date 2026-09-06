//! Sole Sage-backed Exponent-LUT parameter-search integration target.

use std::{env, path::PathBuf};
use tracing_subscriber::EnvFilter;

include!("test_utils.rs");

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_exponent_lut=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

#[test]
#[ignore = "CPU/Sage symbolic parameter search; not GPU runtime"]
fn test_refresh_parameter_search() {
    install_tracing();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sparse_lwr_estimator.sage.py");
    let mut config = SearchConfig::from_env().expect("valid Exponent-LUT refresh profile");
    config.pbc_max_attempts = 128;
    let checkpoint = env::var_os(PHASE1_CHECKPOINT_ENV).map(PathBuf::from);
    // Reviewed tuples carry the exact result of the pinned estimator run.
    // Keep Sage inputs lazy: the normal reviewed search never needs an
    // estimator process, while an evidence-free custom tuple still requires
    // the explicitly configured external estimator.
    let mut estimator_inputs = None;
    let selected = load_or_search_phase1(&config, checkpoint.as_deref(), |u, h, q, p| {
        let tuple = config
            .sparse_lwr_phase1_grid
            .iter()
            .find(|t| t.nu == u && t.h == h && t.q_l == q && t.p == p)
            .ok_or_else(|| "estimator callback received undeclared tuple".to_owned())?;
        if let Some(security_bits) = pinned_estimator_security_bits(tuple)? {
            return Ok(security_bits);
        }

        if estimator_inputs.is_none() {
            let estimator_root = env::var_os("MXX_LATTICE_ESTIMATOR_ROOT")
                .map(PathBuf::from)
                .ok_or_else(|| {
                    "MXX_LATTICE_ESTIMATOR_ROOT must point to pinned checkout for an evidence-free Phase-1 tuple"
                        .to_owned()
                })?;
            let sage_binary = env::var_os("MXX_SAGE_BINARY")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("sage"));
            estimator_inputs = Some((estimator_root, sage_binary));
        }
        let (estimator_root, sage_binary) = estimator_inputs
            .as_ref()
            .expect("estimator inputs initialized for evidence-free tuple");
        let report = run_estimator(
            &tuple.candidate()?,
            &script,
            estimator_root,
            sage_binary,
            Some(&config.phase1_estimator_commit),
        )?;
        if report.cost_model != config.phase1_estimator_cost_model ||
            report.shape_model != config.phase1_estimator_shape_model
        {
            return Err("estimator result disagrees with the configured cost or shape model".to_owned());
        }
        Ok(report.minimum_classical_bits.floor() as u64)
    })
    .expect("Phase-1 search must select a profile");
    let profile_results = search_all_qualified_profiles(&selected, |profile| {
        config.sparse_lwr_universe = profile.universe;
        config.sparse_lwr_weight = profile.weight;
        config.sparse_lwr_modulus = profile.q_l;
        config.sparse_lwr_output_modulus = profile.p;
        search_with_hooks(
            &config,
            &profile,
            |candidate| prepare_candidate(&config, candidate),
            |candidate| {
                if config.bgg_security_envelope.is_some() {
                    return reviewed_bgg_security_bits(&config, candidate);
                }
                let n = 1usize
                    .checked_shl(candidate.log_ring_dimension as u32)
                    .ok_or_else(|| "ring dimension overflow".to_owned())?;
                let dcrt = DCRTPolyParams::new(
                    n as u32,
                    candidate.crt_depth,
                    candidate.crt_bits,
                    candidate.base_bits,
                    None,
                );
                lattice_security_bits(n, dcrt.modulus().as_ref(), config.error_sigma)
            },
            |prepared| check_refresh_bundle(&config, prepared.candidate, prepared),
        )
    })
    .expect("Phase-2 search must produce candidates or canonical no-candidate rows");
    let (selected, result) = select_best_profile_result(profile_results.clone())
        .expect("Phase-2 search must produce a candidate");
    if let Some(path) = checkpoint.as_deref() {
        let evidence = if config.noise_model == NoiseSearchModel::WorstCase {
            Some(
                Phase2CertificationEvidence::certify(&config, &selected, result.candidate)
                    .expect("certify exact persisted Phase-2 profile"),
            )
        } else {
            None
        };
        persist_accepted_phase2_profile(
            path,
            &config,
            &selected,
            result.candidate,
            evidence.as_ref(),
        )
        .expect("persist Phase-2 profile");
    }
    config.crt_bits = result.candidate.crt_bits;
    config.base_bits = result.candidate.base_bits;
    let report = serde_json::to_string_pretty(&search_report_with_profiles(
        &config,
        &selected,
        result,
        profile_results,
    ))
    .unwrap();
    if let Some(path) = env::var_os("MXX_EXPONENT_LUT_REFRESH_SEARCH_OUTPUT") {
        fs::write(path, report).expect("write search report");
    } else {
        println!("{report}");
    }
}
