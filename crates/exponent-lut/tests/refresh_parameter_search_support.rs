//! Sole Sage-backed Exponent-LUT parameter-search integration target.

use std::{env, path::PathBuf};
use tracing_subscriber::EnvFilter;

include!("../src/parameter_search_test_support.rs");

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_exponent_lut=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

#[test]
#[ignore = "CPU/Sage symbolic parameter search; not GPU runtime"]
fn test_refresh_parameter_search() {
    install_tracing();
    let estimator_root = env::var_os("MXX_LATTICE_ESTIMATOR_ROOT")
        .map(PathBuf::from)
        .expect("MXX_LATTICE_ESTIMATOR_ROOT must point to pinned checkout");
    let sage_binary =
        env::var_os("MXX_SAGE_BINARY").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("sage"));
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("sparse_lwr_estimator.sage.py");
    let mut config = SearchConfig::from_env().expect("valid Exponent-LUT refresh profile");
    config.pbc_max_attempts = 128;
    let checkpoint = env::var_os(PHASE1_CHECKPOINT_ENV).map(PathBuf::from);
    let selected = load_or_search_phase1(&config, checkpoint.as_deref(), |u, h, q, p| {
        let tuple = config
            .sparse_lwr_phase1_grid
            .iter()
            .find(|t| t.nu == u && t.h == h && t.q_l == q && t.p == p)
            .ok_or_else(|| "estimator callback received undeclared tuple".to_owned())?;
        let report = run_estimator(
            &tuple.candidate()?,
            &script,
            &estimator_root,
            &sage_binary,
            Some(&config.phase1_estimator_commit),
        )?;
        if (report.minimum_classical_bits - tuple.estimator_minimum_classical_bits).abs() > 1e-12 ||
            !report.minimum_classical_bits.is_finite()
        {
            return Err("estimator result disagrees with declared finite evidence".to_owned());
        }
        Ok(report.minimum_classical_bits.floor() as u64)
    })
    .expect("Phase-1 search must select a profile");
    let (selected, result) = search_qualified_profiles(&selected, |profile| {
        config.sparse_lwr_universe = profile.universe;
        config.sparse_lwr_weight = profile.weight;
        config.sparse_lwr_modulus = profile.q_l;
        config.sparse_lwr_output_modulus = profile.p;
        config.lut_width = profile.tuple.lut_width;
        search_with_hooks(
            &config,
            &profile,
            |candidate| prepare_candidate(&config, candidate),
            |candidate| {
                let n = 1usize
                    .checked_shl(candidate.log_ring_dimension as u32)
                    .ok_or_else(|| "ring dimension overflow".to_owned())?;
                let dcrt = DCRTPolyParams::new(
                    n as u32,
                    candidate.crt_depth,
                    candidate.crt_bits,
                    candidate.base_bits,
                );
                lattice_security_bits(n, dcrt.modulus().as_ref(), config.error_sigma)
            },
            |prepared| check_refresh_bundle(&config, prepared.candidate, prepared),
        )
    })
    .expect("Phase-2 search must produce a candidate");
    if let Some(path) = checkpoint.as_deref() {
        persist_accepted_phase2_profile(path, &config, &selected, result.candidate)
            .expect("persist Phase-2 profile");
    }
    config.crt_bits = result.candidate.crt_bits;
    config.base_bits = result.candidate.base_bits;
    let report = serde_json::to_string_pretty(&search_report(&config, &selected, result)).unwrap();
    if let Some(path) = env::var_os("MXX_EXPONENT_LUT_REFRESH_SEARCH_OUTPUT") {
        fs::write(path, report).expect("write search report");
    } else {
        println!("{report}");
    }
}
