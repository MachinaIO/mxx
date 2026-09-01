//! Symbolic Section-7 Power-LUT refresh parameter search.
//!
//! This is deliberately ignored: it invokes Sage's `lattice-estimator-cli`
//! and constructs large symbolic candidates, but does not execute a GPU/CPU
//! backend, sample runtime artifacts, perform an encryption/decryption round
//! trip, or run a benchmark.  The production integration gate remains the
//! generic operational-noise checker over the actual refresh graph.
//!
//! Set `MXX_POWER_LUT_REFRESH_PHASE1_CHECKPOINT` to an explicit JSON path to
//! persist the public Phase-1 rows and reuse them on a later run. Existing
//! checkpoints are accepted only when their declared grid and security model
//! exactly match this test; without the variable every run starts fresh.

mod refresh_parameter_search_support;
#[path = "sparse_lwr_parameter_support.rs"]
mod sparse_lwr_parameter_support;

use mxx_primitives::poly::PolyParams;
use refresh_parameter_search_support::{
    PHASE1_CHECKPOINT_ENV, SearchConfig, check_refresh_bundle, lattice_security_bits,
    load_or_search_phase1, prepare_candidate, search_report, search_with_hooks,
    sparse_lwr_security_bits,
};
use sparse_lwr_parameter_support::{
    Assessment, AssessmentStatus, DirectLwrStatus, SparseLwrCandidate, run_estimator,
};
use std::{env, path::PathBuf};
use tracing_subscriber::EnvFilter;

fn install_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("mxx_power_lut=debug,mxx_runtime=debug,info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
}

/// First select the smallest qualifying `nu` in the configured Phase-1 grid;
/// then search CRT depth first and ring dimension second within the configured
/// Phase-2 grid, stopping at the first DCRT candidate satisfying security and
/// the generic graph-check hook.
#[test]
#[ignore = "CPU/Sage symbolic parameter search; not GPU runtime"]
fn test_refresh_parameter_search() {
    install_tracing();
    let candidate_a = SparseLwrCandidate::candidate_a().expect("Candidate A must validate");
    let estimator_root = env::var_os("MXX_LATTICE_ESTIMATOR_ROOT")
        .map(PathBuf::from)
        .expect("MXX_LATTICE_ESTIMATOR_ROOT must point to the pinned lattice-estimator checkout");
    let sage_binary =
        env::var_os("MXX_SAGE_BINARY").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("sage"));
    let estimator_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("sparse_lwr_estimator.sage.py");
    let estimator = run_estimator(
        &candidate_a,
        &estimator_script,
        &estimator_root,
        &sage_binary,
        env::var("MXX_LATTICE_ESTIMATOR_COMMIT").ok().as_deref(),
    )
    .expect("Candidate A estimator assessment must succeed");
    let assessment = Assessment::from_estimator(candidate_a.clone(), estimator)
        .expect("Candidate A assessment must validate");
    assert!(assessment.passes_lwe_floor(), "Candidate A must meet the 128-bit LWE floor");
    assert_eq!(assessment.status, AssessmentStatus::IncompleteAttackCoverage);
    assert_eq!(assessment.attack_coverage.direct_lwr, DirectLwrStatus::NotEvaluated);

    let mut config = SearchConfig::from_env().expect("valid configured Power-LUT refresh profile");
    let derived_a = candidate_a.derived().expect("Candidate A derived parameters");
    config.sparse_lwr_universe = candidate_a.nu;
    config.sparse_lwr_weight = candidate_a.h;
    config.sparse_lwr_universe_grid = vec![candidate_a.nu];
    config.sparse_lwr_modulus = candidate_a.q;
    config.sparse_lwr_output_modulus = candidate_a.p;
    config.lut_width = derived_a.w_mod;
    config.pbc_max_attempts = 128;
    let checkpoint_path = env::var_os(PHASE1_CHECKPOINT_ENV).map(PathBuf::from);
    let sparse_profile =
        load_or_search_phase1(&config, checkpoint_path.as_deref(), |universe, weight, q_l, p| {
            sparse_lwr_security_bits(universe, weight, q_l, p)
        })
        .expect("configured sparse-LWR Phase-1 search must select a profile");
    config.sparse_lwr_universe = sparse_profile.universe;
    let result = search_with_hooks(
        &config,
        &sparse_profile,
        |candidate| prepare_candidate(&config, candidate),
        |candidate| {
            let ring_dimension = 1usize
                .checked_shl(candidate.log_ring_dimension as u32)
                .ok_or_else(|| "ring dimension shift overflow".to_owned())?;
            let dcrt = mxx_primitives::poly::dcrt::params::DCRTPolyParams::new(
                ring_dimension as u32,
                candidate.crt_depth,
                config.crt_bits,
                config.base_bits,
            );
            let bgg =
                lattice_security_bits(ring_dimension, dcrt.modulus().as_ref(), config.error_sigma)?;
            Ok(bgg)
        },
        |prepared| check_refresh_bundle(&config, prepared.candidate, prepared),
    )
    .expect("configured Power-LUT refresh search must produce a valid candidate");
    let report =
        serde_json::to_string_pretty(&search_report(&config, &sparse_profile, result)).unwrap();
    if let Some(path) = env::var_os("MXX_POWER_LUT_REFRESH_SEARCH_OUTPUT") {
        std::fs::write(&path, &report).expect("write Power-LUT refresh search report");
    } else {
        println!("{report}");
    }
}
